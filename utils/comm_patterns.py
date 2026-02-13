# -*- coding: utf-8 -*-
import abc

from utils.comm import (
    get_world_size,
    get_inter_node_process_group,
    get_intra_node_process_group,
    get_global_rank,
)
from torch.distributed.fsdp._init_utils import (
    _get_default_group,
)
import torch


def make_tensor(size_mb, dtype=torch.bfloat16):
    element_size = torch.tensor(0, dtype=dtype).element_size()
    size = round(1024**2 * size_mb / element_size)
    return torch.zeros(size, dtype=dtype, device="cuda")


class CommunicationPattern(abc.ABC):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        self.message_size_mb = message_size_mb
        self.dtype = dtype

    @property
    @abc.abstractmethod
    def name(self):
        """The name of the pattern."""

    @abc.abstractmethod
    def execute(self):
        """Run the communication pattern once."""


class AllReduce(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        self.tensor = make_tensor(message_size_mb, dtype=dtype)

    @property
    def name(self):
        return "global_all_reduce"

    def execute(self):
        torch.distributed.all_reduce(self.tensor)


class AllGather(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        input_size = message_size_mb / get_world_size()
        self.input_tensor = make_tensor(input_size, dtype=dtype)
        output_size = self.input_tensor.numel() * get_world_size()
        self.output_tensor = torch.empty(output_size, dtype=dtype, device="cuda")

    @property
    def name(self):
        return "global_all_gather"

    def execute(self):
        torch.distributed.all_gather_into_tensor(self.output_tensor, self.input_tensor)


class ReduceScatter(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):

        super().__init__(message_size_mb, dtype=dtype)
        output_size = message_size_mb / get_world_size()
        self.output_tensor = make_tensor(output_size, dtype=dtype)
        input_size = self.output_tensor.numel() * get_world_size()
        self.input_tensor = torch.zeros(input_size, dtype=dtype, device="cuda")

    @property
    def name(self):
        return "global_reduce_scatter"

    def execute(self):
        torch.distributed.reduce_scatter_tensor(self.output_tensor, self.input_tensor)


class InterNodeAllGather(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        self.pg = get_inter_node_process_group()
        num_gpus = get_intra_node_process_group().size()
        input_size = message_size_mb / num_gpus / self.pg.size()
        self.input_tensor = make_tensor(input_size, dtype=dtype)
        output_size = self.input_tensor.numel() * self.pg.size()
        self.output_tensor = torch.empty(output_size, dtype=dtype, device="cuda")

    @property
    def name(self):
        return "inter_node_all_gather"

    def execute(self):
        torch.distributed.all_gather_into_tensor(
            self.output_tensor, self.input_tensor, group=self.pg
        )


class InterNodeReduceScatter(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        self.pg = get_inter_node_process_group()
        num_gpus = get_intra_node_process_group().size()
        output_size = message_size_mb / num_gpus / self.pg.size()
        self.output_tensor = make_tensor(output_size, dtype=dtype)
        input_size = self.output_tensor.numel() * self.pg.size()
        self.input_tensor = torch.zeros(input_size, dtype=dtype, device="cuda")

    @property
    def name(self):
        return "inter_node_reduce_scatter"

    def execute(self):
        torch.distributed.reduce_scatter_tensor(
            self.output_tensor, self.input_tensor, group=self.pg
        )


class IntraNodeAllGather(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        self.pg = get_intra_node_process_group()
        input_size = message_size_mb / get_world_size()
        self.input_tensor = make_tensor(input_size, dtype=dtype)
        output_size = self.input_tensor.numel() * self.pg.size()
        self.output_tensor = torch.empty(output_size, dtype=dtype, device="cuda")

    @property
    def name(self):
        return "intra_node_all_gather"

    def execute(self):
        torch.distributed.all_gather_into_tensor(
            self.output_tensor, self.input_tensor, group=self.pg
        )


class IntraNodeReduceScatter(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        self.pg = get_intra_node_process_group()
        output_size = message_size_mb / get_world_size()
        self.output_tensor = make_tensor(output_size, dtype=dtype)
        input_size = self.output_tensor.numel() * self.pg.size()
        self.input_tensor = torch.zeros(input_size, dtype=dtype, device="cuda")

    @property
    def name(self):
        return "intra_node_reduce_scatter"

    def execute(self):
        torch.distributed.reduce_scatter_tensor(
            self.output_tensor, self.input_tensor, group=self.pg
        )


class HybridReduceScatter(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        self.local_pg = get_intra_node_process_group()
        self.cross_node_pg = get_inter_node_process_group()
        output_size = message_size_mb / self.local_pg.size()
        self.output_tensor = make_tensor(output_size, dtype=dtype)
        input_size = self.output_tensor.numel() * self.local_pg.size()
        self.input_tensor = torch.zeros(input_size, dtype=dtype, device="cuda")

    @property
    def name(self):
        return "hybrid_reduce_scatter"

    def execute(self):
        torch.distributed.reduce_scatter_tensor(
            self.output_tensor,
            self.input_tensor,
            group=self.local_pg,
        )
        torch.distributed.all_reduce(self.output_tensor, group=self.cross_node_pg)


class Zero_2_5(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        self.local_pg = get_intra_node_process_group()
        self.cross_node_pg = get_inter_node_process_group()
        output_size_local_rs = message_size_mb / self.local_pg.size()
        self.output_tensor_local_rs = make_tensor(output_size_local_rs, dtype=dtype)
        input_size = self.output_tensor_local_rs.numel() * self.local_pg.size()
        self.input_tensor = torch.zeros(input_size, dtype=dtype, device="cuda")
        output_size_cross_rs = output_size_local_rs / self.cross_node_pg.size()
        self.output_tensor_cross_rs = make_tensor(output_size_cross_rs, dtype=dtype)
        self.output_tensor_cross_ag = make_tensor(output_size_local_rs, dtype=dtype)

    @property
    def name(self):
        return "zero_2_5"

    def execute(self):
        torch.distributed.reduce_scatter_tensor(
            self.output_tensor_local_rs,
            self.input_tensor,
            group=self.local_pg,
        )
        torch.distributed.reduce_scatter_tensor(
            self.output_tensor_cross_rs,
            self.output_tensor_local_rs,
            group=self.cross_node_pg,
        )
        torch.distributed.all_gather_into_tensor(
            self.output_tensor_cross_ag,
            self.output_tensor_cross_rs,
            group=self.cross_node_pg,
        )


class ReduceScatter2Hop(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        self.local_pg = get_intra_node_process_group()
        self.cross_node_pg = get_inter_node_process_group()
        output_size_local_rs = message_size_mb / self.local_pg.size()
        self.output_tensor_local_rs = make_tensor(output_size_local_rs, dtype=dtype)
        input_size = self.output_tensor_local_rs.numel() * self.local_pg.size()
        self.input_tensor = torch.zeros(input_size, dtype=dtype, device="cuda")
        output_size_cross_rs = output_size_local_rs / self.cross_node_pg.size()
        self.output_tensor_cross_rs = make_tensor(output_size_cross_rs, dtype=dtype)

    @property
    def name(self):
        return "reduce_scatter_2hop"

    def execute(self):
        torch.distributed.reduce_scatter_tensor(
            self.output_tensor_local_rs,
            self.input_tensor,
            group=self.local_pg,
        )
        torch.distributed.reduce_scatter_tensor(
            self.output_tensor_cross_rs,
            self.output_tensor_local_rs,
            group=self.cross_node_pg,
        )


class Zero_2_5_Global(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)

        self.rs_inputs = []
        rs_output_size = message_size_mb / get_world_size()
        # need to rearrange order of grads to send results to right worker
        for i in range(get_world_size()):
            self.rs_inputs.append(make_tensor(rs_output_size, dtype=dtype))

        self.output_tensor_rs = make_tensor(rs_output_size, dtype=dtype)
        ag_output_size = self.rs_inputs[0].numel() * get_world_size()
        self.output_tensor_ag = torch.zeros(ag_output_size, dtype=dtype, device="cuda")

    @property
    def name(self):
        return "zero_2_5_global"

    def execute(self):
        torch.distributed.reduce_scatter(
            self.output_tensor_rs,
            self.rs_inputs,
        )
        torch.distributed.all_gather_into_tensor(
            self.output_tensor_ag, self.output_tensor_rs
        )


class AllGatherUneven(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        self.pg = _get_default_group()

        self.ag_outputs = []
        szes = [1, 1, 1500000 * message_size_mb]
        # szes = [500000 * message_size_mb, 500000 * message_size_mb, 500000 * message_size_mb]
        # szes = [750000*message_size_mb, 1, 750000*message_size_mb]
        # szes = [500000*message_size_mb, 1, 1000000*message_size_mb]

        for i in range(get_world_size()):
            sz = szes[i]
            self.ag_outputs.append(torch.ones(sz, dtype=dtype, device="cuda"))
        self.ag_input = torch.zeros(szes[get_global_rank()], dtype=dtype, device="cuda")

    @property
    def name(self):
        return "all_gather_uneven"

    def execute(self):
        self.pg.allgather(
            self.ag_outputs,
            self.ag_input,
        )


class ReduceScatterUneven(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        self.pg = _get_default_group()

        self.rs_inputs = []
        # szes = [1, 1, 1500000 * message_size_mb]
        # szes = [500000 * message_size_mb, 500000 * message_size_mb, 500000 * message_size_mb]
        # szes = [750000*message_size_mb, 1, 750000*message_size_mb]
        # szes = [500000 * message_size_mb, 1, 1000000 * message_size_mb]
        szes = [
            500000 * message_size_mb,
            500000 * message_size_mb,
            1000000 * message_size_mb,
        ]

        for i in range(get_world_size()):
            sz = szes[i]
            self.rs_inputs.append(torch.ones(sz, dtype=dtype, device="cuda"))

        self.rs_output = torch.zeros(
            szes[get_global_rank()], dtype=dtype, device="cuda"
        )

    @property
    def name(self):
        return "reduce_scatter_uneven"

    def execute(self):
        self.pg.reduce_scatter(
            self.rs_output,
            self.rs_inputs,
        )


class Broadcast(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        self.pg = _get_default_group()
        self.input = torch.zeros(message_size_mb * 1000000, dtype=dtype, device="cuda")

    @property
    def name(self):
        return "broadcast"

    def execute(self):
        torch.distributed.broadcast(self.input, 1, group=self.pg)


class AllGatherUnevenRatio(CommunicationPattern):
    def __init__(self, message_size_mb, shard_ratio, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        self.pg = _get_default_group()

        self.ag_outputs = []
        scale_factor = (1024 * 1024) // torch.tensor([], dtype=dtype).element_size()
        total_message_size = scale_factor * message_size_mb

        # atleast one per gpu
        szes = [max(1, int(ratio * total_message_size)) for ratio in shard_ratio]

        for sz in szes:
            self.ag_outputs.append(torch.ones(sz, dtype=dtype, device="cuda"))
        self.ag_input = torch.zeros(szes[get_global_rank()], dtype=dtype, device="cuda")

    @property
    def name(self):
        return "all_gather_uneven"

    def execute(self):
        self.pg.allgather(self.ag_outputs, self.ag_input)


class ReduceScatterUnevenRatio(CommunicationPattern):
    def __init__(self, message_size_mb, shard_ratio, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        self.pg = _get_default_group()

        self.rs_inputs = []
        scale_factor = (1024 * 1024) // torch.tensor([], dtype=dtype).element_size()
        total_message_size = scale_factor * message_size_mb

        szes = [max(1, int(ratio * total_message_size)) for ratio in shard_ratio]

        for i in range(get_world_size()):
            sz = szes[i]
            self.rs_inputs.append(torch.ones(sz, dtype=dtype, device="cuda"))

        self.rs_output = torch.zeros(
            szes[get_global_rank()], dtype=dtype, device="cuda"
        )

    @property
    def name(self):
        return "reduce_scatter_uneven"

    def execute(self):
        self.pg.reduce_scatter(
            self.rs_output,
            self.rs_inputs,
        )
