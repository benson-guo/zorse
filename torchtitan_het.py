import contextlib
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.cuda import Event
from models.hub import get_model, is_vision_model
from models.parallelize_model import parallelize_model
from models.pipeline_model import pipeline_model
from utils.argparser_utils import parse_args
from utils.data_loader import get_dataset
from utils.parallelism import ParallelDims
from utils.logger import get_logger, init_logger

from utils.profile import get_profiler_context, print_memory_stats
from torch.distributed.tensor._api import DTensor
from utils.train_utils import get_profiler_path, print_metrics


def get_train_context(
    enable_loss_parallel: bool = False, enable_compiled_autograd: bool = False
):
    """
    Create a training context manager with optional features.
    """

    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())
            if enable_compiled_autograd:
                stack.enter_context(
                    torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                )
            yield

    return context


# see here: https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/experimental/__init__.py#L15
@contextlib.contextmanager
def implicit_replication():
    """
    This context manager allows :class:`DTensor` to implicitly treat all non-DTensors (``torch.Tensor``)
    in the program be replicate :class:`DTensor` s during the operator computation.

    .. warning:: This might possible lead to incorrect results if ``torch.Tensor`` s are not replicated
        in practice, please use it at your discretion.
    """
    try:
        DTensor._op_dispatcher._allow_implicit_replication = True
        yield
    finally:
        DTensor._op_dispatcher._allow_implicit_replication = False


def train(
    args,
    model,
    optimizer,
    loss_fn,
    data_iter,
    train_context,
    pp_schedule,
    pp_mesh,
    parallel_dims,
    profiler_path,
    warmup_iterations,
    iterations,
    print_mem_step,
    skip_profile,
    device,
):
    logger = get_logger()
    start_event = Event(enable_timing=True)
    end_event = Event(enable_timing=True)

    total_iterations = warmup_iterations + iterations

    if args.offload_activations:
        offload_ctx = torch.autograd.graph.save_on_cpu(pin_memory=True)
    else:
        offload_ctx = contextlib.nullcontext()

    max_allocated_mem = 0  # Initialize variable to store max allocated memory

    for step_idx in range(total_iterations):
        if step_idx == warmup_iterations:
            dist.barrier()
            start_event.record()

        # Profiling context for specific iterations
        if (
            step_idx
            in range(warmup_iterations - args.profiling_iterations, warmup_iterations)
            and not skip_profile
        ):
            out_dir = os.path.join(profiler_path, f"iteration_{step_idx}")
            profiler_ctx = get_profiler_context(out_dir=out_dir, unique_gpus_only=True)
        else:
            profiler_ctx = contextlib.nullcontext()

        # Fetch the next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(get_dataset(args))
            batch = next(data_iter)

        input_ids, labels = batch

        # Move input_ids to device if this is the first pipeline stage
        if not pp_mesh or pp_mesh.get_local_rank() == 0:
            input_ids = input_ids.to(device)

        # Record iteration start time
        iteration_start = Event(enable_timing=True)
        iteration_end = Event(enable_timing=True)
        iteration_start.record()

        with profiler_ctx:
            if step_idx == print_mem_step:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            if step_idx == print_mem_step:
                print_memory_stats("pre-forward", all_ranks=False)

            # Determine if this is the last pipeline stage
            is_last_stage = (
                not pp_mesh or pp_mesh.get_local_rank() == pp_mesh.size() - 1
            )

            for ga_idx in range(args.gradient_accumulation_steps):
                with torch.autocast(
                    device_type="cuda", dtype=args.autocast_dtype
                ), offload_ctx, implicit_replication():  #
                    if parallel_dims.pp_enabled:
                        if pp_mesh.get_local_rank() == 0:
                            pp_schedule.step(input_ids)
                        elif is_last_stage:
                            losses = []
                            # Move labels to the current device (last pipeline stage)
                            labels = labels.to(device)
                            outputs = pp_schedule.step(target=labels, losses=losses)
                        else:
                            pp_schedule.step()

                        # Accumulate losses across pipeline microbatches
                        loss = (
                            torch.mean(torch.stack(losses))
                            if is_last_stage
                            else torch.Tensor([-1.0]).to(device)
                        )
                    else:
                        # input_ids = DTensor.from_local(input_ids, mesh=pp_mesh, dim=0, placements=[Replicate()])
                        # labels = DTensor.from_local(labels, mesh=pp_mesh, dim=0, placements=[Replicate()])

                        outputs = model(input_ids, labels=labels)
                        if hasattr(outputs, "loss"):
                            loss = outputs.loss
                        else:
                            loss = outputs.sum()
                        loss.backward()

                if step_idx == print_mem_step and ga_idx == 0:
                    max_allocated_mem = print_memory_stats(
                        "post-backward", all_ranks=True
                    )["max_allocated"]

                if (
                    not args.optimizer_in_backwards
                    and ga_idx == args.gradient_accumulation_steps - 1
                ):
                    # This context manager allows :class:`DTensor` to implicitly treat all non-DTensors (``torch.Tensor``)
                    # in the program be replicate :class:`DTensor` s during the operator computation
                    with implicit_replication():
                        optimizer.step()
                        optimizer.zero_grad()

                if step_idx == print_mem_step and ga_idx == 0:
                    print_memory_stats("post-optimize", all_ranks=False)

        # Record iteration end time and synchronize
        iteration_end.record()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Compute elapsed time
        elapsed_time = iteration_start.elapsed_time(iteration_end)

        # Log loss and iteration time
        if parallel_dims.pp_enabled:
            # Only the last stage has the actual loss
            current_loss = loss.item() if is_last_stage else -1.0
        else:
            current_loss = loss.item()

        logger.info(
            f"Iteration {step_idx} Loss {current_loss} Time {elapsed_time:.2f} ms"
        )

    # Record end event and synchronize
    end_event.record()
    torch.cuda.synchronize()
    avg_iteration_time = start_event.elapsed_time(end_event) / iterations

    # Print final metrics
    print_metrics(args, max_allocated_mem, avg_iteration_time, profiler_path)


from torch.distributed.pipelining.stage import _PipelineStageBase

forward_one_chunk_orig = _PipelineStageBase.forward_one_chunk


# patch forward_one_chunk so we replace outputs for the last stage
# with dummy values, since they are not needed for traning and
# consume a lot of memory (batch size x sequence length x vocab size)
def forward_one_chunk_patch(
    self,
    fwd_chunk_id: int,
    args,
    kwargs,
):
    res = forward_one_chunk_orig(self, fwd_chunk_id, args, kwargs)
    if self.is_last:
        self.output_chunks[-1] = torch.tensor([0.0])
    return res


def main():
    args = parse_args()
    if args.compile_transformer_blocks:
        # u3anand: setting this for better performance with compile
        torch.set_float32_matmul_precision("high")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    print(f"Using device: {device}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    # Initialize distributed environment
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.world_size = world_size
    args.global_batch_size = (
        args.batch_size
        * args.world_size
        // args.tt_pp
        // args.tt_tp
        * args.gradient_accumulation_steps
    )
    args.local_batch_size = args.batch_size
    init_logger()
    logger = get_logger()
    parallel_dims = ParallelDims(
        dp_replicate=args.tt_dp_replicate,
        dp_shard=args.tt_dp_shard,
        tp=args.tt_tp,
        pp=args.tt_pp,
        world_size=world_size,
        enable_loss_parallel=False,
    )

    def sequence_loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1), labels.flatten(0, 1)
        )

    def image_loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(pred.mean(dim=1), labels)

    if is_vision_model(args.model_name):
        loss_fn = image_loss_fn
    else:
        loss_fn = sequence_loss_fn

    if args.use_deepcopy_for_build or not parallel_dims.pp_enabled:
        if is_vision_model(args.model_name):
            model = get_model(
                args.model_name,
                image_size=args.image_size,
                layers=args.model_layer_override,
            )
        else:
            model = get_model(
                args.model_name,
                vocab_size=args.vocab_size,
                seq_length=args.seq_length,
                layers=args.model_layer_override,
            )
    else:
        model = None  # Model will be built per stage in pipeline_model()

    data = get_dataset(args)
    data_iter = iter(data)
    # Build device mesh
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    logger.info(f"World mesh: {world_mesh}, DP Degree: {dp_degree}, DP Rank: {dp_rank}")

    # Apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]
        torch.distributed.pipelining.stage._PipelineStageBase.forward_one_chunk = (
            forward_one_chunk_patch
        )
        # Apply Pipeline Parallelism
        pp_schedule, model_parts = pipeline_model(
            model, pp_mesh, parallel_dims, device, loss_fn, args
        )

        # Apply FSDP
        for m in model_parts:
            parallelize_model(m, world_mesh, parallel_dims, args)
            m.to(device)
            m.train()

        parameters = []
        for m in model_parts:
            parameters.extend(list(m.parameters()))
        optimizer = torch.optim.Adam(
            parameters,
            lr=0.0001,
            fused=not (args.no_fused_optimizer or args.cpu_offload),
        )
    else:
        pp_mesh = None
        parallelize_model(model, world_mesh, parallel_dims, args)
        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.0001,
            fused=not (args.no_fused_optimizer or args.cpu_offload),
        )

    # Set up training context
    train_context = get_train_context(
        enable_loss_parallel=False,
        enable_compiled_autograd=False,
    )

    # Set up profiling and logging
    profiler_path = get_profiler_path(args)
    warmup_iterations = args.warmup_iterations
    iterations = args.iterations
    print_mem_step = warmup_iterations - 3
    skip_profile = args.skip_profile

    # Start training
    train(
        args=args,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        data_iter=data_iter,
        train_context=train_context,
        pp_schedule=pp_schedule if parallel_dims.pp_enabled else None,
        pp_mesh=pp_mesh,
        parallel_dims=parallel_dims,
        profiler_path=profiler_path,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        print_mem_step=print_mem_step,
        skip_profile=skip_profile,
        device=device,
    )

    # Clean up distributed environment
    dist.destroy_process_group()


if __name__ == "__main__":
    main()