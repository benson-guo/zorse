# -*- coding: utf-8 -*-
import logging
import os
from typing import Optional
from utils.comm import get_global_rank


class PipelineLogger:
    def __init__(
        self,
        log_dir: str = "pipeline_logs",
        enabled: bool = True,
        log_level: str = logging.INFO,
    ):
        self.enabled = enabled
        if not enabled:
            return
        self.global_rank = get_global_rank()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"pipeline_rank_{self.global_rank}.log")
        self.setup_logger(log_level=log_level)
        self.current_step = 0

    def setup_logger(self, log_level: str = logging.INFO):
        self.logger = logging.getLogger(f"pipeline_rank_{self.global_rank}")
        self.logger.setLevel(log_level)

        # don't log stdout
        self.logger.propagate = False
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # file for each rank
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        fh.setFormatter(formatter)

        # Add only file handler to logger
        self.logger.addHandler(fh)

    def set_step(self, step: int):
        if not self.enabled:
            return
        self.current_step = step
        self.log_separator()
        self.logger.info(f"Starting Step {step}")
        self.log_separator()

    def log_separator(self):
        self.logger.info("-" * 80)

    def get_log_prefix(self, stage_id: int, mb_idx: int):
        return f"Step {self.current_step} - Stage {stage_id} - MB {mb_idx} - Rank {self.global_rank}:"

    def debug(self, stage_id: int, mb_idx: int, msg: str):
        if not self.enabled:
            return
        self.logger.debug(self.get_log_prefix(stage_id, mb_idx) + " " + msg)

    def log_forward_recv(self, stage_id: int, mb_idx: int, src_rank: Optional[int]):
        if not self.enabled:
            return
        msg = (
            self.get_log_prefix(stage_id, mb_idx)
            + f" RECV activation from Rank {src_rank}"
        )
        self.logger.info(msg)

    def log_forward_compute(self, stage_id: int, mb_idx: int, layer_idx: int):
        if not self.enabled:
            return
        msg = (
            self.get_log_prefix(stage_id, mb_idx)
            + f" COMPUTE forward Layer {layer_idx}"
        )
        self.logger.info(msg)

    def log_forward_send(self, stage_id: int, mb_idx: int, dst_rank: Optional[int]):
        if not self.enabled:
            return
        msg = (
            self.get_log_prefix(stage_id, mb_idx)
            + f" SEND activation to Rank {dst_rank}"
        )
        self.logger.info(msg)

    def log_backward_recv(self, stage_id: int, mb_idx: int, src_rank: Optional[int]):
        if not self.enabled:
            return
        msg = (
            self.get_log_prefix(stage_id, mb_idx)
            + f" RECV gradient from Rank {src_rank}"
        )
        self.logger.info(msg)

    def log_backward_compute(self, stage_id: int, mb_idx: int, layer_idx: int):
        if not self.enabled:
            return
        msg = (
            self.get_log_prefix(stage_id, mb_idx)
            + f" COMPUTE backward Layer {layer_idx}"
        )
        self.logger.info(msg)

    def log_backward_send(self, stage_id: int, mb_idx: int, dst_rank: Optional[int]):
        if not self.enabled:
            return
        msg = (
            self.get_log_prefix(stage_id, mb_idx) + f" SEND gradient to Rank {dst_rank}"
        )
        self.logger.info(msg)


_PIPELINE_LOGGER = None


def init_pipeline_logger(log_dir, log_level: str = logging.INFO) -> PipelineLogger:
    """Initialize the global pipeline logger."""
    global _PIPELINE_LOGGER
    _PIPELINE_LOGGER = PipelineLogger(log_dir=log_dir, log_level=log_level)
    return _PIPELINE_LOGGER


def get_pipeline_logger(
    log_dir: str = "pipeline_logs", log_level: str = logging.INFO
) -> PipelineLogger:
    """Get the global pipeline logger instance."""
    global _PIPELINE_LOGGER
    if _PIPELINE_LOGGER is None:
        _PIPELINE_LOGGER = init_pipeline_logger(log_dir, log_level=log_level)
    return _PIPELINE_LOGGER
