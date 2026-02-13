# -*- coding: utf-8 -*-
from typing import Optional
import torch
from torch import distributed as dist
from grolar_utils.pipeline_comm import StageCommPattern


class PipelineCommunicatorGloo:
    def __init__(
        self,
        stage_id: int,
        num_stages: int,
        curr_rank: int,
        comm_plan: StageCommPattern,
        process_group: dist.ProcessGroup,
        buffer_size: torch.Size,
        dtype: torch.dtype,
    ):
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.curr_rank = curr_rank
        self.comm_plan = comm_plan
        self.process_group = process_group
        if not hasattr(PipelineCommunicatorGloo, "_shared_mem_copy_stream"):
            PipelineCommunicatorGloo._shared_mem_copy_stream_send = torch.cuda.Stream()
        self.mem_copy_stream_send = (
            PipelineCommunicatorGloo._shared_mem_copy_stream_send
        )
        if not hasattr(PipelineCommunicatorGloo, "_shared_mem_copy_stream_recv"):
            PipelineCommunicatorGloo._shared_mem_copy_stream_recv = torch.cuda.Stream()
        self.mem_copy_stream_recv = (
            PipelineCommunicatorGloo._shared_mem_copy_stream_recv
        )
        self.num_mbs = len(comm_plan.fwd_send_to_ranks)
        self.dtype = dtype

        # CPU buffers for send/recv operations
        self.fwd_send_buffers = {}  # mb_idx -> buffer
        self.bwd_send_buffers = {}
        self.fwd_recv_buffers = {}
        self.bwd_recv_buffers = {}

        # Initialize buffers for all possible microbatches
        for mb_idx in range(self.num_mbs):
            if self.comm_plan.fwd_recv_from_ranks[mb_idx] is not None:
                self.fwd_recv_buffers[mb_idx] = torch.empty(
                    buffer_size, device="cpu", pin_memory=True, dtype=self.dtype
                )

            if self.comm_plan.bwd_recv_from_ranks[mb_idx] is not None:
                self.bwd_recv_buffers[mb_idx] = torch.empty(
                    buffer_size, device="cpu", pin_memory=True, dtype=self.dtype
                )

    def set_mem_copy_stream(self, stream: torch.cuda.Stream):
        self.mem_copy_stream = stream

    def wait_for_stream(self, stream: Optional[torch.cuda.Stream] = None):
        if stream is None:
            stream = torch.cuda.current_stream()
        self._shared_mem_copy_stream_send.wait_stream(stream)

    def send_forward(self, tensor: torch.Tensor, mb_idx: int) -> Optional[dist.Work]:
        dst_rank = self.comm_plan.fwd_send_to_ranks[mb_idx]
        if dst_rank is None:
            return None

        # Reuse or create CPU buffer for this microbatch
        if mb_idx not in self.fwd_send_buffers:
            self.fwd_send_buffers[mb_idx] = torch.empty_like(
                tensor, device="cpu", pin_memory=True, dtype=self.dtype
            )

        cpu_buffer = self.fwd_send_buffers[mb_idx]

        with torch.cuda.stream(self._shared_mem_copy_stream_send):
            cpu_buffer.copy_(tensor)
            tag = self.comm_plan.fwd_send_tags[mb_idx]
            work = dist.isend(cpu_buffer, dst_rank, group=self.process_group, tag=tag)
        return work

    def recv_forward(self, tensor_size: torch.Size, mb_idx: int) -> Optional[dist.Work]:
        src_rank = self.comm_plan.fwd_recv_from_ranks[mb_idx]
        if src_rank is None:
            return None

        cpu_buffer = self.fwd_recv_buffers[mb_idx]
        gpu_buffer = torch.empty(
            tensor_size, device="cuda", dtype=self.dtype, requires_grad=True
        )

        tag = self.comm_plan.fwd_recv_tags[mb_idx]

        # Async receive into CPU buffer with tag
        work = dist.irecv(cpu_buffer, src_rank, group=self.process_group, tag=tag)

        return WorkWrapper(work, [cpu_buffer], [gpu_buffer], self.mem_copy_stream_recv)

    def send_backward(self, tensor: torch.Tensor, mb_idx: int) -> Optional[dist.Work]:
        dst_rank = self.comm_plan.bwd_send_to_ranks[mb_idx]
        if dst_rank is None:
            return None

        # Reuse or create CPU buffer
        if mb_idx not in self.bwd_send_buffers:
            self.bwd_send_buffers[mb_idx] = torch.empty_like(
                tensor, device="cpu", pin_memory=True, dtype=self.dtype
            )

        cpu_buffer = self.bwd_send_buffers[mb_idx]

        with torch.cuda.stream(self._shared_mem_copy_stream_send):
            cpu_buffer.copy_(tensor)
            tag = self.comm_plan.bwd_send_tags[mb_idx]
            work = dist.isend(cpu_buffer, dst_rank, group=self.process_group, tag=tag)
        return work

    def recv_backward(
        self, tensor_size: torch.Size, mb_idx: int
    ) -> Optional[dist.Work]:
        src_rank = self.comm_plan.bwd_recv_from_ranks[mb_idx]
        if src_rank is None:
            return None

        cpu_buffer = self.bwd_recv_buffers[mb_idx]
        gpu_buffer = torch.empty(
            tensor_size, device="cuda", dtype=self.dtype, requires_grad=True
        )

        tag = self.comm_plan.bwd_recv_tags[mb_idx]

        work = dist.irecv(cpu_buffer, src_rank, group=self.process_group, tag=tag)

        return WorkWrapper(work, [cpu_buffer], [gpu_buffer], self.mem_copy_stream_recv)


class WorkWrapper:
    def __init__(
        self,
        work: dist.Work,
        cpu_buffers: list,
        gpu_buffers: list,
        stream: torch.cuda.Stream,
    ):
        self.work = work
        self.cpu_buffers = cpu_buffers
        self.gpu_buffers = gpu_buffers
        self.stream = stream

    def wait(self) -> None:
        self.work.wait()

        with torch.cuda.stream(self.stream):
            for gpu_buf, cpu_buf in zip(self.gpu_buffers, self.cpu_buffers):
                gpu_buf.requires_grad_(False)  # cannot copy a grad tensor
                gpu_buf.copy_(cpu_buf)
                gpu_buf.requires_grad_(True)

    def result(self):
        return self.gpu_buffers
