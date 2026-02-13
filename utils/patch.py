# -*- coding: utf-8 -*-
import math
import torch
import contextlib
from typing import Any, Callable, Dict, no_type_check, Optional, Tuple
import torch.nn as nn
from torch.distributed.fsdp.api import ShardingStrategy, BackwardPrefetch
import torch.distributed as dist
from torch.distributed.utils import _p_assert, _free_storage, _cast_forward_inputs
from torch.distributed.fsdp._common_utils import (
    TrainingState,
    _FSDPState,
    _no_dispatch_record_stream,
)
from torch.distributed.fsdp._runtime_utils import (
    _post_backward_reshard,
    _reduce_grad,
    _low_precision_hook_enabled,
    _div_if_needed,
    _accumulate_sharded_grad,
    _post_reduce_grad_callback,
    _is_composable,
    _reset_flat_param_grad_info_if_needed,
    _assert_in_training_states,
    _PrefetchMode,
    _get_training_state,
    _prefetch_handle,
    _get_reduce_scatter_tensors,
    _register_pre_backward_hooks,
    _register_post_backward_reshard_only_hook,
    _root_pre_forward,
    _register_post_backward_hook,
)
from torch.distributed.fsdp._flat_param import (
    HandleShardingStrategy,
    HandleTrainingState,
    FlatParamHandle,
)
from utils.global_state import get_split_state, set_split_state, set_split_state_nested
from utils.logger import get_logger

logger = get_logger()
# Global for post backward hook (used inside sync)
_post_backward_hook = torch.distributed.fsdp._runtime_utils._post_backward_hook


@torch.no_grad()
def _post_backward_hook_sync(
    state: _FSDPState,
    handle: FlatParamHandle,
    *unused: Any,
):
    # sync communication with compute stream
    if state.sharding_strategy == ShardingStrategy.SHARD_GRAD_OP:
        tensor = torch.randint(low=0, high=1, size=(1,), device="cuda")
        dist.broadcast(tensor, 0, group=state.process_group)
    # original code
    _post_backward_hook(state, handle, *unused)


def _offload_model_params(handle):
    split_state = get_split_state()
    memcpy_stream = split_state["memcpy_stream"]
    with torch.cuda.stream(memcpy_stream):
        flat_param = handle.flat_param
        if not hasattr(flat_param, "cpu_local_shard"):
            flat_param.cpu_local_shard = torch.empty_like(
                flat_param.data, device="cpu", pin_memory=True
            )
        flat_param.cpu_local_shard.copy_(flat_param.data, non_blocking=True)
        flat_param.data = torch.empty(0, device=flat_param.device)
        # flat_param._local_shard points to flat_param.data, so need to clear reference to offloaded data
        flat_param._local_shard = flat_param.data


def _post_reduce_grad_callback_patch(
    state: _FSDPState,
    handle: FlatParamHandle,
    # Additional arguments needed for the callback logic
    grad: torch.Tensor,
):
    # original code
    # _offload_grad(state, handle, grad_to_offload)
    # _post_backward_use_sharded_grad_views(handle)

    # apply optimizer update immidiately so no need to offload grad
    # requires use_orig_params=False since use_orig_params=True not compatible with some of our models
    assert not handle._use_orig_params

    split_state = get_split_state()
    flat_param = handle.flat_param
    if flat_param._accumulated_grad_count < split_state["microbatches"]:
        return

    # assumes that if we are not applying optimizer in backwards, there is also no offloading
    if "optimizer" not in state._comm_hook_state:
        return

    optimizer_stream = split_state["optimizer_stream"]
    optimizer_stream.wait_stream(state._post_backward_stream)
    # run optimizer in dedicated stream so doesn't block reduce scatter or backwards computation
    with torch.cuda.stream(optimizer_stream):
        # apply optimizer update
        optimizer = state._comm_hook_state["optimizer"]

        # restore optimizer params
        fsdp_module_params = state.parameters()
        param_groups = optimizer.param_groups
        for param, param_group in zip(fsdp_module_params, param_groups):
            with torch.no_grad():
                optimizer_param = param_group["params"][0]
                param_shard_tensor = param.data
                optimizer_param.data = param_shard_tensor

        # maybe cast gradients during mixed precision training
        grad_data_type = grad.dtype
        param_data_type = param_groups[0]["params"][0].dtype
        if grad_data_type != param_data_type:
            grad.data = grad.data.to(param_data_type)

        # extract slice of gradients for each parameter
        offset = 0
        for param_group in param_groups:
            param = param_group["params"][0]
            param_size = param.numel()
            if handle._offload_params:
                grad_slice = handle.flat_param._cpu_grad.copy_(
                    grad[offset : offset + param_size].detach(), non_blocking=False
                )
                _no_dispatch_record_stream(grad.data, state._post_backward_stream)
            else:
                grad_slice = grad[offset : offset + param_size]
            param.grad = grad_slice
            offset += param_size

        # optimizer step
        optimizer.step()
        optimizer.zero_grad()
        del flat_param.grad
        del grad
        del flat_param._saved_grad_shard
        if handle._offload_params:
            flat_param._cpu_grad = None

        split_state = get_split_state()
        if split_state.get("offload_model_params", False):
            memcpy_stream = split_state["memcpy_stream"]
            memcpy_stream.wait_stream(optimizer_stream)
            _offload_model_params(handle)
            # clear optimizer references to tensors
            # otherwise flat_param.data will not be freed, because optimizer still has references to it
            for param_group in optimizer.param_groups:
                param = param_group["params"][0]
                param.data = torch.empty(0, device=flat_param.device)


def prepare_gradient_for_optim_noop(self):
    """
    Patched prepare_gradient_for_optim_fused_opt that skips gradient post-postprocessing for
    the fused optimizer. Behaviour is unchanged for fsdp modules that do not use the fused optimizer.
    """
    return


@torch.no_grad()
def _post_backward_hook_patch(
    state: _FSDPState,
    handle: FlatParamHandle,
    *unused: Any,
):
    flat_param = handle.flat_param
    flat_param._post_backward_called = True

    logger.debug("running custom post backward hook")
    with torch.autograd.profiler.record_function(
        "FullyShardedDataParallel._post_backward_hook"
    ):
        _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
        # For multiple applications of reentrant AC across submodules sharing
        # the same `FlatParameter`, the post-backward hook may run multiple
        # times in one backward, in which case we permit the state to already
        # be in `BACKWARD_POST`.
        _p_assert(
            handle._training_state
            in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.BACKWARD_POST),
            f"Expects `BACKWARD_PRE` or `BACKWARD_POST` state but got {handle._training_state}",
        )
        handle._training_state = HandleTrainingState.BACKWARD_POST

        if flat_param.grad is None:
            return
        if flat_param.grad.requires_grad:
            raise RuntimeError("FSDP does not support gradients of gradients")

        _post_backward_reshard(state, handle)

        if not state._sync_gradients:
            if handle._use_orig_params:
                handle._use_unsharded_grad_views()
            return

        split_state = get_split_state()
        gradient_accumulation = split_state["microbatches"] > 1
        if not hasattr(flat_param, "_accumulated_grad_count"):
            flat_param._accumulated_grad_count = 0
        flat_param._accumulated_grad_count += 1

        # maybe accumulate gradient
        if gradient_accumulation:
            logger.debug(
                f"Gradient accumulation enabled, count: {flat_param._accumulated_grad_count} / {split_state['microbatches']}"
            )
            if hasattr(flat_param, "_accumulated_grad"):
                flat_param._accumulated_grad += flat_param.grad
            else:
                flat_param._accumulated_grad = flat_param.grad
            if flat_param._accumulated_grad_count < split_state["microbatches"]:
                del flat_param.grad
                return

        # Wait for all ops in the current stream (e.g. gradient computation) to
        # finish before reduce-scattering the gradient
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            state._post_backward_stream.wait_stream(
                state._device_handle.current_stream()
            )

        with state._device_handle.stream(state._post_backward_stream):
            autograd_computed_grad = flat_param.grad.data
            if (
                not _low_precision_hook_enabled(state)
                and flat_param.grad.dtype != handle._reduce_dtype
                # If we are forcing full precision but communicating grads
                # (i.e. model.eval() + full precision in eval was configured), don't downcast gradient.
                and not handle._force_full_precision
            ):
                if gradient_accumulation:
                    flat_param._accumulated_grad = flat_param._accumulated_grad.to(
                        handle._reduce_dtype
                    )
                else:
                    flat_param.grad.data = flat_param.grad.to(handle._reduce_dtype)

            # add support for uneven sharding
            if hasattr(handle, "uneven_shard") and handle.uneven_shard:
                split_idx = split_state["split_idx_map"][handle]
                if gradient_accumulation:
                    unsharded_grad = flat_param._accumulated_grad
                    flat_param._accumulated_grad = None
                else:
                    unsharded_grad = flat_param.grad.data
                flat_param.grad = None

                rs_output_size = split_idx[handle.rank + 1] - split_idx[handle.rank]
                rs_output = torch.empty(
                    rs_output_size,
                    dtype=unsharded_grad.dtype,
                    device=unsharded_grad.device,
                )

                _div_if_needed(unsharded_grad, state._gradient_predivide_factor)
                rs_inputs = []
                for i in range(len(split_idx) - 1):
                    rs_input = unsharded_grad[split_idx[i] : split_idx[i + 1]]
                    rs_inputs.append(rs_input)

                dist.reduce_scatter(
                    rs_output,
                    rs_inputs,
                    group=state.process_group,
                )

                _div_if_needed(rs_output, state._gradient_postdivide_factor)
                grad_to_offload = _accumulate_sharded_grad(state, handle, rs_output)
                _post_reduce_grad_callback(state, handle, grad_to_offload)
            else:
                logger.debug("Reducing gradient")
                if handle.uses_sharded_strategy:
                    if gradient_accumulation:
                        logger.debug("_reduce_grad patch")
                        # use flat_param._accumulated_grad instead of flat_param.grad
                        _reduce_grad_patch(
                            state, handle, unsharded_grad=flat_param._accumulated_grad
                        )
                    else:
                        _reduce_grad(state, handle)
                else:
                    logger.debug("_reduce_grad_no_shard")
                    _reduce_grad_no_shard_patch(
                        state, handle, unsharded_grad=flat_param._accumulated_grad
                    )
            # Since the unsharded gradient is produced in the computation
            # stream and consumed in the post-backward stream, inform the
            # caching allocator (before it goes out of scope)
            _no_dispatch_record_stream(
                autograd_computed_grad, state._post_backward_stream
            )

            if hasattr(flat_param, "_accumulated_grad"):
                del flat_param._accumulated_grad


# -------------------------------------------------------
#   Patches for Sharding/Unsharding
# -------------------------------------------------------


@torch.no_grad()
def shard_patch(self):
    flat_param = self.flat_param
    if not self.uses_sharded_strategy:
        self._init_shard_metadata(0, 0, flat_param.numel() - 1)
    else:
        _p_assert(
            flat_param.storage_offset() == 0,
            "The `FlatParameter` is not the sole occupant of its storage",
        )
        split_state = get_split_state()

        # track even splits and use fsdp _get_shard
        split_layer_even = False
        numel = flat_param.numel()
        num_gpus = self.world_size
        min_shard_size = num_gpus
        # compute shard sizes
        if split_state["split_uneven"]:
            if split_state["proportional_split"]:
                # split proportional to ratios
                shard_sizes = [
                    max(min_shard_size, int(numel * split_state["model_partitions"][i]))
                    for i in range(num_gpus)
                ]
            else:
                # greedy strategy to maximize even splits
                total_params = split_state["total_model_params"]
                mask = split_state["rank_mask"]
                ratio = split_state["shard_ratio"]
                shard_sizes = [0 for _ in range(num_gpus)]
                remaining_params = numel

                if sum(mask) < num_gpus:
                    leftover_ratio = [
                        split_state["model_partitions"][i] - ratio[i]
                        for i in range(num_gpus)
                    ]
                    shard_sizes = [
                        max(
                            min_shard_size,
                            int(numel * leftover_ratio[i] / sum(leftover_ratio)),
                        )
                        for i in range(num_gpus)
                    ]
                else:
                    while remaining_params > 0 and any(mask):
                        curr_world_size = sum(mask)

                        if curr_world_size == 1:
                            split_params = remaining_params  # case: 1 gpu left
                        else:
                            min_ratio = min(
                                [
                                    (split_state["model_partitions"][i] - ratio[i])
                                    for i in range(num_gpus)
                                    if mask[i]
                                ]
                            )
                            split_params = min(
                                min_ratio * total_params * curr_world_size,
                                remaining_params,
                            )

                        # split layer evenly use pytorch _get_shard
                        if (
                            split_params == remaining_params
                            and curr_world_size == num_gpus
                        ):
                            split_layer_even = True

                        if split_params <= 0:
                            break

                        # Distribute split_params across active GPUs
                        distributed_params = 0
                        for i in range(num_gpus):
                            if mask[i]:
                                gpu_share = math.ceil(split_params / curr_world_size)
                                shard_sizes[i] += gpu_share
                                distributed_params += gpu_share
                                ratio[i] += gpu_share / total_params
                                if ratio[i] >= split_state["model_partitions"][i]:
                                    mask[i] = 0

                        remaining_params -= distributed_params

                # Update the state
                set_split_state(key="shard_ratio", value=ratio)
                set_split_state(key="rank_mask", value=mask)

        if split_layer_even or not split_state["split_uneven"]:
            self.uneven_shard = False
            sharded_flat_param, numel_padded = FlatParamHandle._get_shard(
                flat_param, self.rank, self.world_size
            )
            flat_param.set_(sharded_flat_param)  # type: ignore[call-overload]
            start_idx = sharded_flat_param.numel() * self.rank
            end_idx = sharded_flat_param.numel() * (self.rank + 1) - 1  # inclusive

        else:
            self.uneven_shard = True
            # Calculate split_idx based on updated shard_sizes
            split_idx = [0]
            for shard_idx, size in enumerate(shard_sizes):
                next_idx = split_idx[-1] + size
                # due to precision errors we might end up with empty shards,
                # potentially adjust indices to avoid this issue
                num_shards_behind = len(shard_sizes) - 1 - shard_idx
                next_idx = min(next_idx, numel - num_shards_behind * min_shard_size)
                split_idx.append(next_idx)
            split_idx[-1] = numel

            set_split_state_nested(key="split_idx_map", subkey=self, value=split_idx)

            max_shard_size = max(shard_sizes)

            flat_param._max_shard_size = max_shard_size
            flat_param._uneven_unshard_size = numel

            numel_padded = 0
            start_idx = split_idx[self.rank]
            end_idx = split_idx[self.rank + 1] - 1  # inclusive
            sharded_flat_param = flat_param[start_idx : end_idx + 1].clone()
            flat_param.set_(sharded_flat_param)  # type: ignore[call-overload]

        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            allocated = flat_param._typed_storage()._size() > 0
            if allocated:
                flat_param._typed_storage()._resize_(0)
        self._init_shard_metadata(numel_padded, start_idx, end_idx)

    if self._use_orig_params:
        self._use_sharded_views()


def unshard_patch(self):
    logger.debug("unsharding layer")
    split_state = get_split_state()
    stream_ctx = contextlib.nullcontext()
    gradient_accumulation = split_state["gradient_accumulation"]
    unshard_in_compute = split_state["unshard_in_compute"]
    if gradient_accumulation and not unshard_in_compute:
        # run in new stream so allgather doesn't block the backward pass
        # originally, the backwards for the split() operation called in _use_unsharded_flat_param
        # was being executed in the unshard stream, which was blocked by allgather
        new_unshard_stream = get_split_state()["streams"][-1]
        stream_ctx = torch.cuda.stream(new_unshard_stream)

    if not self.needs_unshard():
        # Even when not needing an unshard, we should switch to using
        # the unsharded flat parameter
        # print("dont need unshard")
        with stream_ctx:
            unsharded_flat_param = (
                self._get_padded_unsharded_flat_param()
                if self.uses_sharded_strategy
                else self.flat_param
            )
            self._use_unsharded_flat_param(unsharded_flat_param)
        return

    if hasattr(self, "uneven_shard") and self.uneven_shard:
        self.flat_param._padded_unsharded_size = torch.Size(
            [self.flat_param._uneven_unshard_size]
        )
        if (
            self.flat_param._full_param_padded.shape
            != self.flat_param._padded_unsharded_size
        ):
            self.flat_param._full_param_padded = torch.empty(
                self.flat_param._uneven_unshard_size,
                device=self.flat_param.device,
                dtype=self.flat_param.dtype,
            )
            _free_storage(self.flat_param._full_param_padded)

        unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
        split_idx = split_state["split_idx_map"][self]
        ag_input = self.flat_param.data
        ag_outputs = []

        for i in range(len(split_idx) - 1):
            ag_outputs.append(unsharded_flat_param[split_idx[i] : split_idx[i + 1]])
        dist.all_gather(
            ag_outputs,
            ag_input,
            self.process_group,
        )

        padded_unsharded_flat_param = unsharded_flat_param
    else:
        unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
        padded_unsharded_flat_param = self._all_gather_flat_param(unsharded_flat_param)

    with stream_ctx:
        self._use_unsharded_flat_param(padded_unsharded_flat_param)


# -------------------------------------------------------
#   Patches for Gradient Accumulation
# -------------------------------------------------------


def reshard_patch(self, free_unsharded_flat_param: bool):
    """
    Runs the reshard logic. This includes freeing the unsharded flat
    parameter if ``free_unsharded_flat_param`` and switching to using the
    sharded flat parameter. Note that this also implicitly offloads
    the sharded flat parameter (if CPU offload is enabled) by pointing
    it to the ``_local_shard`` attribute which resides on CPU.
    """
    # Switch to the sharded `FlatParameter` before freeing to prevent
    # "use-after-free"-type bugs with external profiling tools, where for
    # `use_orig_params=True`, the `param` does not point to valid memory
    # when setting `param.data = ...` in `_use_sharded_views()`.

    # modified to skip reshard until last microbatch during gradient accumulation
    split_state = get_split_state()
    microbatches = split_state["microbatches"]
    gradient_accumulation = split_state["gradient_accumulation"]

    in_backwards = (
        self._training_state == HandleTrainingState.BACKWARD_POST
        or self._training_state == HandleTrainingState.BACKWARD_PRE
    )
    flat_param = self.flat_param
    grad_count = (
        flat_param._accumulated_grad_count
        if hasattr(flat_param, "_accumulated_grad_count")
        else 0
    )
    reshard_microbatch = grad_count == microbatches - 1
    if gradient_accumulation and in_backwards and not reshard_microbatch:
        logger.debug(
            f"Don't reshard GA: {gradient_accumulation} Backwards: {in_backwards} Reshard Microbatch: {reshard_microbatch} Grad Count: {grad_count}"
        )
        # should only reshard after the last microbatch
        return False

    self._use_sharded_flat_param()
    if free_unsharded_flat_param:
        self._free_unsharded_flat_param()
    return True


def _reduce_grad_patch(
    state: _FSDPState, handle: FlatParamHandle, unsharded_grad=None
) -> None:
    """
    For sharded strategies, this runs gradient reduction, sharded gradient
    accumulation if needed, and the post-reduction callback.
    """
    split_state = get_split_state()
    post_reduce_grad_fn = (
        _post_reduce_grad_callback_patch
        if split_state["optimizer_in_backwards"]
        else _post_reduce_grad_callback
    )
    flat_param = handle.flat_param
    uses_hybrid_sharded_strategy = handle._sharding_strategy in (
        HandleShardingStrategy.HYBRID_SHARD,
        HandleShardingStrategy._HYBRID_SHARD_ZERO2,
    )
    # We clear `.grad` to permit multiple backwards. This avoids a race where
    # the second backward pass computation precedes ahead of the first backward
    # pass reduction, which is possible since the reduction is issued in a
    # separate stream and is async and would result in reducing the wrong
    # gradient.
    unsharded_grad = flat_param.grad.data if unsharded_grad is None else unsharded_grad
    flat_param.grad = None
    padded_unsharded_grad, new_sharded_grad = _get_reduce_scatter_tensors(
        state, unsharded_grad
    )
    if not split_state["skip_reduce_scatter"]:
        if state._comm_hook is None:  # default path
            _div_if_needed(padded_unsharded_grad, state._gradient_predivide_factor)
            dist.reduce_scatter_tensor(
                new_sharded_grad,
                padded_unsharded_grad,
                group=state.process_group,
            )
            if uses_hybrid_sharded_strategy:
                state._all_reduce_stream.wait_stream(state._post_backward_stream)
                with state._device_handle.stream(state._all_reduce_stream):
                    # Since the new sharded gradient is produced in the post-
                    # backward stream and consumed in the all-reduce stream,
                    # inform the caching allocator
                    _no_dispatch_record_stream(
                        new_sharded_grad, state._all_reduce_stream
                    )
                    dist.all_reduce(new_sharded_grad, group=state._inter_node_pg)
                    _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
                    grad_to_offload = _accumulate_sharded_grad(
                        state, handle, new_sharded_grad
                    )
                    post_reduce_grad_fn(state, handle, grad_to_offload)
                    return
            _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
        else:
            state._comm_hook(
                state._comm_hook_state, padded_unsharded_grad, new_sharded_grad
            )
            # NOTE: HSDP variants do not support communication hook.
    grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
    post_reduce_grad_fn(state, handle, grad_to_offload)


orig_post_forward_reshard = torch.distributed.fsdp._runtime_utils._post_forward_reshard


# Patch such that when hybrid sharding with process_group.size() = 1,
# we perform all gather on inter node process group. This is equivalent to DP
# This function is called since FSDP will change sharding strategy to NO_SHARD
def _reduce_grad_no_shard_patch(
    state: _FSDPState, handle: FlatParamHandle, unsharded_grad=None
) -> None:
    """
    For no-shard, this runs gradient reduction (which directly covers any
    gradient accumulation implicitly) and the post-reduction callback.
    """
    # CHANGE: We use post_reduce_grad_callback_patch if optimizer_in_backwards
    split_state = get_split_state()
    post_reduce_grad_fn = (
        _post_reduce_grad_callback_patch
        if split_state["optimizer_in_backwards"]
        else _post_reduce_grad_callback
    )
    flat_param = handle.flat_param
    unsharded_grad = flat_param.grad.data if unsharded_grad is None else unsharded_grad
    flat_param.grad = None
    if state._comm_hook is None:  # default path
        _div_if_needed(unsharded_grad, state._gradient_predivide_factor)
        # PATCH
        if hasattr(state, "_inter_node_pg") and state._inter_node_pg.size() > 1:
            dist.all_reduce(unsharded_grad, group=state._inter_node_pg)
        else:
            if state.process_group.size() > 1:
                dist.all_reduce(unsharded_grad, group=state.process_group)
        _div_if_needed(unsharded_grad, state._gradient_postdivide_factor)
    else:
        state._comm_hook(state._comm_hook_state, unsharded_grad)
    # For `NO_SHARD`, we can keep the low precision gradients by simply
    # omitting the cast altogether
    # CHANGE: replace _cast_grad_to_param_dtype with _accumulate_sharded_grad, so state is consistent with _reduce_grad_patch()
    # if not handle._keep_low_precision_grads:
    #     _cast_grad_to_param_dtype(state, flat_param.grad, flat_param)
    grad_to_offload = _accumulate_sharded_grad(state, handle, unsharded_grad)
    post_reduce_grad_fn(state, handle, grad_to_offload)


def _post_forward_reshard_patch(state, handle):
    if not handle:
        return
    logger.debug("Resharding after forwards")
    # only reshard if params are sharded across > 1 GPU
    free_unsharded_flat_param = state.process_group.size() > 1
    return _reshard_patch(state, handle, free_unsharded_flat_param)


@no_type_check
def _pre_forward_patch(
    state: _FSDPState,
    handle: Optional[FlatParamHandle],
    unshard_fn: Callable,
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    in_backwards: bool = False,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Runs the pre-forward logic. This includes an opportunity to unshard
    currently sharded parameters such as those for the current forward and
    registering post-backward hooks for these current parameters. This function
    also converts forward ``args`` and ``kwargs`` to the given precision.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        unshard_fn (Optional[Callable]): A callable to unshard any currently
            sharded parameters or ``None`` to not do any unsharding.
        module (nn.Module): Module whose forward this method runs right before;
            expected by the hook signature.
        args (Tuple[Any, ...]): Module forward ``args``.
        kwargs (Dict[str, Any]): Module forward ``kwargs``.
    """
    with torch.profiler.record_function("FullyShardedDataParallel._pre_forward"):
        # For `fully_shard` + `checkpoint`, skip pre-forward logic in the
        # recomputed forward
        if handle and handle._training_state == HandleTrainingState.BACKWARD_PRE:
            # For both checkpoint implementations, we do not need to re-cast
            # inputs here since they will be checkpointed in the low precision
            # either by AC or normally by autograd as long as the AC region is
            # nested within FSDP
            return args, kwargs
        state.training_state = TrainingState.FORWARD_BACKWARD
        state._exec_order_data.record_pre_forward(handle, module.training)
        if handle:
            handle._training_state = HandleTrainingState.FORWARD
        if unshard_fn is not None:
            unshard_fn(state, handle)
        # Register post-backward hooks to reshard the parameters and reduce-scatter
        # their gradients. They must be re-registered every forward pass in case
        # the `grad_fn` is mutated.
        ### CHANGE STARTS HERE
        # Benson: We only register post backward hook if we are doing forwards recomputation
        # in backwards pass. Otherwise it gets registered on the first forwards pass,
        # and there seems to be some bug that prevents gradients from being propogated
        # https://github.com/pytorch/pytorch/issues/94857
        # For zero3 use default behavior
        split_state = get_split_state()
        if split_state["zero2_pipeline"]:
            if in_backwards:
                _register_post_backward_hook(state, handle)
        else:
            _register_post_backward_hook(state, handle)
        ### CHANGE ENDS HERE
        # We have to reallocate the _cpu_grad if optimizer overlap
        # set the grad to None in the backward pass.
        if handle and handle._offload_params and handle.flat_param._cpu_grad is None:
            handle.flat_param._cpu_grad = torch.zeros_like(
                handle.flat_param._local_shard, device=torch.device("cpu")
            ).pin_memory(device=state.compute_device)

        should_cast_forward_inputs = (
            state._handle and not state._handle._force_full_precision
        )

        if should_cast_forward_inputs and state.mixed_precision.cast_forward_inputs:
            # Recursively convert args and kwargs to specified precision.
            input_dtype: Optional[torch.dtype] = state.mixed_precision.param_dtype
            args, kwargs = _cast_forward_inputs(input_dtype, *args, **kwargs)
        _register_post_backward_reshard_only_hook(state, handle, args, kwargs)
        return args, kwargs


def forward_patch(self, *args, **kwargs):
    """
    Runs the forward pass for the wrapped module, inserting FSDP-specific
    pre- and post-forward sharding logic.
    """
    is_first_microbatch = kwargs.get("is_first_microbatch", False)
    is_last_microbatch = kwargs.get("is_last_microbatch", False)
    skip_reshard = kwargs.get("skip_reshard", False)
    in_backwards = kwargs.get("in_backwards", False)
    del kwargs["is_first_microbatch"]
    del kwargs["is_last_microbatch"]
    del kwargs["skip_reshard"]
    del kwargs["in_backwards"]

    handle = self._handle
    with torch.autograd.profiler.record_function("FullyShardedDataParallel.forward"):
        ### CHANGE STARTS HERE: Save in _prefetched state, since we may have prefetched it
        # but _root_pre_forward sets _prefetched to False
        already_prefetched = self._handle._prefetched
        args, kwargs = _root_pre_forward(self, self, args, kwargs)
        self._handle._prefetched = already_prefetched
        ### CHANGE ENDS HERE
        unused = None
        logger.debug(
            f"Calling pre forward, first microbatch: {is_first_microbatch}, last microbatch: {is_last_microbatch}, skip reshard: {skip_reshard}, in_backwards: {in_backwards}"
        )
        wait_unshard = is_first_microbatch
        args, kwargs = _pre_forward_patch(
            self,
            handle,
            lambda state, handle: _pre_forward_unshard_patch(
                state,
                handle,
                wait_unshard=wait_unshard,
                is_first_microbatch=is_first_microbatch,
                in_backwards=in_backwards,
            ),
            self._fsdp_wrapped_module,
            args,
            kwargs,
            in_backwards=in_backwards,
        )
        output = self._fsdp_wrapped_module(*args, **kwargs)

        reshard_fn = (
            _post_forward_reshard_patch
            if (is_last_microbatch and not skip_reshard and not in_backwards)
            else None
        )
        return _post_forward_patch(
            self, handle, reshard_fn, self, unused, output, in_backwards=in_backwards
        )


def _post_forward_patch(
    state: _FSDPState,
    handle: Optional[FlatParamHandle],
    reshard_fn,
    module,
    input: Any,
    output: Any,
    in_backwards: bool = False,
) -> Any:
    """
    Runs the post-forward logic. This includes an opportunity to reshard
    currently unsharded parameters such as those used in the current forward
    and registering pre-backward hooks on the forward outputs.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        reshard_fn (Optional[Callable]): A callable to reshard any currently
            unsharded parameters (e.g. from the current forward) or ``None`` to
            not do any resharding.
        module (nn.Module): Module whose forward just ran, which should be a
            fully sharded module (see [Note: Fully Sharded Module]); expected
            by the hook signature.
        input (Any): Unused; expected by the hook signature.
        output (Any): Forward pass output; pre-backward hooks are registered on
            the tensors that require gradients in this output.

    Postcondition: Each ``FlatParameter`` 's data points to the sharded flat
    parameter.
    """
    # patch so we register the pre-backward hook on the forward during backward pass
    with torch.profiler.record_function("FullyShardedDataParallel._post_forward"):
        # For `fully_shard` + `checkpoint`, skip post-forward logic in the
        # recomputed forward
        # comment out this case since we are not auto checkpointing
        # if (
        #     handle
        #     and handle._training_state == HandleTrainingState.BACKWARD_PRE
        # ):
        #     print(f"skip post forward {handle._training_state}")
        #     return output
        state._exec_order_data.record_post_forward(handle)
        if reshard_fn is not None:
            reshard_fn(state, handle)
        if not in_backwards:
            return output
        logger.debug(f"register pre backward hook {torch.is_grad_enabled()}")
        # Register pre-backward hooks to unshard the flat parameters for the
        # gradient computation (if needed)
        output = _register_pre_backward_hooks(state, module, output, handle)
        state.training_state = TrainingState.IDLE
        if handle:
            handle._training_state = HandleTrainingState.IDLE
        return output


def _pre_forward_unshard_patch(
    state,
    handle,
    wait_unshard=True,
    is_first_microbatch=False,
    in_backwards=False,
) -> None:
    """Unshards parameters in the pre-forward."""
    if not handle:
        return
    # If the handles have been prefetched, then there is no need to call
    # `_unshard()` again

    split_state = get_split_state()
    # patch to wait for unshard stream only when necessary
    if not handle._prefetched and is_first_microbatch:
        logger.debug("Unshard handle before forwards")
        _unshard_patch(state, handle, state._unshard_stream, state._pre_unshard_stream)
    handle._needs_pre_forward_unshard = False
    if wait_unshard:
        logger.debug("Pre forward wait stream")
        unshard_finished_event = split_state["unshard_events"][handle]
        with torch.profiler.record_function(
            "FullyShardedDataParallel._pre_forward_unshard_wait"
        ):
            state._device_handle.current_stream().wait_event(unshard_finished_event)
    else:
        return
    if in_backwards:
        return
    with torch.profiler.record_function(
        "FullyShardedDataParallel._pre_forward_prefetch_patch"
    ):
        _prefetch_handle(state, handle, _PrefetchMode.FORWARD)


def _check_order_patch(self, handle, is_training):
    pass


def _finalize_params_patch(
    state,
) -> None:
    """Finalizes the parameters before the next iteration."""
    split_state = get_split_state()
    handle = state._handle
    if not handle:
        return
    flat_param = handle.flat_param
    microbatches = split_state["microbatches"]
    logger.debug(f"Finalize params called on hash {hash(flat_param)}")
    is_last_microbatch = flat_param._accumulated_grad_count == microbatches

    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        if hasattr(flat_param, "_post_backward_hook_handle"):
            pbhs_handle = flat_param._post_backward_hook_handle

            if is_last_microbatch:
                pbhs_handle.remove()
                del flat_param._post_backward_hook_handle
    else:
        if hasattr(flat_param, "_post_backward_hook_state"):
            post_backward_hook_state_len = len(flat_param._post_backward_hook_state)
            expected_post_backward_hook_state_len = int(flat_param.requires_grad) + 1
            _p_assert(
                post_backward_hook_state_len == expected_post_backward_hook_state_len,
                f"Invalid: ``_post_backward_hook_state``: {flat_param._post_backward_hook_state}",
            )

            if is_last_microbatch:
                flat_param._accumulated_grad_count = 0
                flat_param._post_backward_hook_state[-1].remove()
                delattr(flat_param, "_post_backward_hook_state")
    if flat_param.requires_grad:
        if not state._sync_gradients:
            # Preserve the gradient accumulation state if not synchronizing
            # gradients: `.grad` remains the unsharded gradient  from prior
            # `no_sync()` iterations, and `_saved_grad_shard` remains the
            # sharded gradient from the last synchronized iteration
            return
        if not split_state["optimizer_in_backwards"] and is_last_microbatch:
            handle.prepare_gradient_for_optim()
        _p_assert(
            hasattr(flat_param, "_post_backward_called"),
            "Expects `_post_backward_called` to be set on the `FlatParameter`",
        )
        flat_param._post_backward_called = False


def _get_handle_to_prefetch_patch(
    state,
    current_handle: FlatParamHandle,
) -> FlatParamHandle:
    """
    Returns a :class:`list` of the handles keys to prefetch for the next
    module(s), where ``current_handle`` represents the current module.

    "Prefetching" refers to running the unshard logic early (without
    synchronization), and the "next" modules depend on the recorded execution
    order and the current training state.
    """
    training_state = _get_training_state(current_handle)
    valid_training_states = (
        HandleTrainingState.BACKWARD_PRE,
        HandleTrainingState.BACKWARD_POST,
        HandleTrainingState.FORWARD,
    )
    _p_assert(
        training_state in valid_training_states,
        f"Prefetching is only supported in {valid_training_states} but "
        f"currently in {training_state}",
    )
    # eod = state._exec_order_data
    target_handle: Optional[FlatParamHandle] = None

    split_state = get_split_state()
    if (
        training_state == HandleTrainingState.BACKWARD_PRE
        and state.backward_prefetch == BackwardPrefetch.BACKWARD_PRE
    ) or (
        training_state == HandleTrainingState.BACKWARD_POST
        and state.backward_prefetch == BackwardPrefetch.BACKWARD_POST
    ):
        # target_handle_candidate = eod.get_handle_to_backward_prefetch(current_handle)
        target_handle_candidate = split_state["backward_prefetch"].get(
            current_handle, None
        )
        if (
            target_handle_candidate
            # since forwards has not registered backwards hook, this will always be false
            # and target_handle_candidate._needs_pre_backward_unshard
            and not target_handle_candidate._prefetched
        ):
            target_handle = target_handle_candidate
        else:
            target_handle = None
    elif training_state == HandleTrainingState.FORWARD and state.forward_prefetch:
        # target_handle_candidate = eod.get_handle_to_forward_prefetch(current_handle)
        target_handle_candidate = split_state["forward_prefetch"].get(
            current_handle, None
        )
        if (
            target_handle_candidate
            and target_handle_candidate._needs_pre_forward_unshard
            and not target_handle_candidate._prefetched
        ):
            target_handle = target_handle_candidate
        else:
            target_handle = None

    if target_handle is not None:
        logger.debug(
            f"Prefetching {target_handle} {training_state} {state.backward_prefetch}"
        )
    else:
        logger.debug(f"Not prefetching {training_state} {state.backward_prefetch}")

    return target_handle


orig_init_streams = torch.distributed.fsdp._runtime_utils._init_streams


def _init_streams_patch(
    state: _FSDPState,
) -> None:
    split_state = get_split_state()
    if split_state["streams"] is not None:
        (
            state._default_stream,
            state._unshard_stream,
            state._post_backward_stream,
            state._pre_unshard_stream,
            state._all_reduce_stream,
            state._new_unshard_stream,
        ) = split_state["streams"]
    else:
        orig_init_streams(state)
        state._new_unshard_stream = state._device_handle.Stream(priority=0)
        set_split_state(
            "streams",
            (
                state._default_stream,
                state._unshard_stream,
                state._post_backward_stream,
                state._pre_unshard_stream,
                state._all_reduce_stream,
                state._new_unshard_stream,
            ),
        )


def _wait_for_computation_stream_patch(stream1, stream2, stream3):
    # do not wait
    pass


def _register_post_backward_final_callback_patch(state, module):
    pass


def _pre_backward_hook_patch(
    state: _FSDPState,
    module,
    handle: FlatParamHandle,
    grad,
    *unused,
):
    """
    Prepares ``_handle`` 's ``FlatParameter`` s for gradient computation.

    Args:
        module (nn.Module): Fully sharded module (see [Note: Fully Sharded
            Module]).
    """
    # Only run the pre-backward hook once per group of handles involved in the
    # same module forward computation
    logger.debug("Pre backward hook called")
    microbatch_count = (
        handle.flat_param._accumulated_grad_count
        if hasattr(handle.flat_param, "_accumulated_grad_count")
        else 0
    )
    if microbatch_count == 0:
        logger.debug(
            f"Pre backward hook prefetching, microbatch count: {microbatch_count}"
        )
        handle._needs_pre_backward_unshard = False
        with torch.profiler.record_function(
            "FullyShardedDataParallel._pre_backward_prefetch_patch"
        ):
            old_state = handle._training_state
            handle._training_state = (
                torch.distributed.fsdp._flat_param.HandleTrainingState.BACKWARD_PRE
            )
            _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)
            handle._training_state = old_state
    if (
        handle
        and hasattr(handle, "_ran_pre_backward_hook")
        and handle._ran_pre_backward_hook
    ):
        return grad

    with torch.profiler.record_function("FullyShardedDataParallel._pre_backward_hook"):
        # Queue the post-backward callback once for the root FSDP instance to
        # attach it to the outermost backward graph task so that it is called
        # after all backward calls complete
        if state._is_root and not state._post_backward_callback_queued:
            _register_post_backward_final_callback_patch(state, module)
            _reset_flat_param_grad_info_if_needed(state._all_handles)
        elif handle:
            allowed_states = [TrainingState.IDLE]
            if _is_composable(state):
                allowed_states.append(TrainingState.FORWARD_BACKWARD)
            _assert_in_training_states(state, allowed_states)
        state.training_state = TrainingState.FORWARD_BACKWARD
        # Queueing the post-backward callback is the only logic that is not
        # per-handle in the pre-backward hook, so we can return early here if
        # there are no handles.
        if not handle:
            return grad
        handle._training_state = HandleTrainingState.BACKWARD_PRE

        # if handle._needs_pre_backward_unshard:
        #     # If the handles have been prefetched, then there is no need to
        #     # call `_unshard()` again
        #     if not handle._prefetched:
        #         _unshard(
        #             state,
        #             handle,
        #             state._unshard_stream,
        #             state._pre_unshard_stream,
        #         )
        #     if is_leader():
        #         print("wait for unshard stream")
        #     # state._device_handle.current_stream().wait_stream(state._unshard_stream)

        # Set this to `False` to ensure that a mistargeted prefetch does not
        # actually unshard these handles
        # handle._needs_pre_backward_unshard = False
        # with torch.profiler.record_function(
        #     "FullyShardedDataParallel._pre_backward_prefetch"
        # ):
        #     _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)

        handle.prepare_gradient_for_backward()
        handle._ran_pre_backward_hook = True
        return grad


def _reshard_patch(
    state: _FSDPState,
    handle: FlatParamHandle,
    free_unsharded_flat_param: bool,
):
    """
    Reshards the handle. ``free_unsharded_flat_param`` indicates whether to
    free the handle's padded unsharded flat parameter.
    """
    ret = handle.reshard(free_unsharded_flat_param)

    split_state = get_split_state()
    training_state = _get_training_state(handle)
    # only offload after _reshard during forwards, since backwards offloading happens in _post_reduce_grad_callback_patch
    offload_model_params = (
        split_state.get("offload_model_params", False)
        and training_state == HandleTrainingState.FORWARD
    )

    if not ret:
        if offload_model_params:
            # wait for computation to finish
            split_state["memcpy_stream"].wait_stream(state._default_stream)
            _offload_model_params(handle)
        return

    logger.debug("Resharding params")
    if state.limit_all_gathers and free_unsharded_flat_param:
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            # We don't run a even queue for freeing under torch compile atm
            # But maybe we need to? TODO(voz): Look into this
            free_event = state._device_handle.Event()
            free_event.record()
            state._free_event_queue.enqueue(free_event)
    handle.post_reshard()
    # Flat parameter freed or not, we always have to "unshard" the parameter
    # upon next access to get its shape correct.
    handle._prefetched = False

    if offload_model_params:
        # wait for computation to finish
        split_state["memcpy_stream"].wait_stream(state._default_stream)
        _offload_model_params(handle)


_orig_unshard = torch.distributed.fsdp._runtime_utils._unshard


@no_type_check
def _unshard_patch(
    state: _FSDPState,
    handle: FlatParamHandle,
    unshard_stream: torch.Stream,
    pre_unshard_stream: torch.Stream,
) -> None:
    """
    Unshards the handles in ``handles``. If the handles are in
    :meth:`summon_full_params` and are using mixed precision, then they are
    forced to full precision.

    Postcondition: Each handle's ``FlatParameter`` 's data is the padded
    unsharded flattened parameter on the compute device.
    """
    split_state = get_split_state()
    unshard_finished_event = split_state["unshard_events"][handle]
    _orig_unshard(state, handle, unshard_stream, pre_unshard_stream)
    unshard_finished_event.record(stream=unshard_stream)


def enable_gradient_accumulation():
    global logger
    logger = get_logger()
    set_split_state(key="gradient_accumulation", value=True)
    FlatParamHandle.shard = shard_patch
    # running unshard on separate stream
    FlatParamHandle.unshard = unshard_patch
    # dont' reshard until last microbatch
    FlatParamHandle.reshard = reshard_patch

    # accumulate grad
    torch.distributed.fsdp._runtime_utils._unshard = _unshard_patch
    torch.distributed.fsdp._runtime_utils._post_backward_hook = (
        _post_backward_hook_patch
    )
    # patches for supporting gradient accumulation
    # patch to reshard on last microbatch of forwards
    torch.distributed.fsdp.FullyShardedDataParallel.forward = forward_patch
    # patch to skip order check, since we follow order of layers
    torch.distributed.fsdp._exec_order_utils._ExecOrderData._check_order = (
        _check_order_patch
    )
    # patch to reset accumulated gradient, hooks, and call prepare_gradient_for_optim on last microbatch
    torch.distributed.fsdp._runtime_utils._finalize_params = _finalize_params_patch
    # patch to support prefetching next layer
    torch.distributed.fsdp._runtime_utils._get_handle_to_prefetch = (
        _get_handle_to_prefetch_patch
    )
    # patch to reuse cuda streams
    torch.distributed.fsdp._runtime_utils._init_streams = _init_streams_patch
    # do not register post backward final callback since it causes backwards to wait on gradient reduction
    # we manually call this after the entire backwards
    torch.distributed.fsdp._runtime_utils._register_post_backward_final_callback = (
        _register_post_backward_final_callback_patch
    )
    torch.distributed.fsdp._runtime_utils._wait_for_computation_stream = (
        _wait_for_computation_stream_patch
    )
    # call prefetch once per layer
    torch.distributed.fsdp._runtime_utils._pre_backward_hook = _pre_backward_hook_patch
    # patch to remove post reshard logic if not resharded
    torch.distributed.fsdp._runtime_utils._reshard = _reshard_patch


__all__ = [
    "_post_backward_hook_sync",
    "_post_reduce_grad_callback_patch",
    "prepare_gradient_for_optim_noop",
    "_post_backward_hook_patch",
    "shard_patch",
    "unshard_patch",
    "reshard_patch",
    "_reduce_grad_patch",
    "_post_forward_reshard_patch",
    "_pre_forward_unshard_patch",
    "_check_order_patch",
    "_finalize_params_patch",
    "_get_handle_to_prefetch_patch",
    "_init_streams_patch",
    "_wait_for_computation_stream_patch",
    "_register_post_backward_final_callback_patch",
    "_pre_backward_hook_patch",
    "enable_gradient_accumulation",
]
