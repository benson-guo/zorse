# -*- coding: utf-8 -*-
import logging
try:
    from utils.comm import is_local_leader
except:
    # needed for flashflex
    from groler.utils.comm import is_local_leader


class LoggerUtility:
    def __init__(self, name=__name__, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.setup_logger(level)

    def setup_logger(self, level):
        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger.setLevel(level)

    def info(self, message, leader_only=True):
        if leader_only and not is_local_leader():
            return
        self.logger.info(message)

    def debug(self, message, leader_only=True):
        if leader_only and not is_local_leader():
            return
        self.logger.debug(message)

    def error(self, message, leader_only=True):
        if leader_only and not is_local_leader():
            return
        self.logger.error(message)


LOGGER = LoggerUtility()


def init_logger(name=__name__, log_level=logging.INFO):
    global LOGGER
    LOGGER = LoggerUtility(name, log_level)


def get_logger():
    global LOGGER
    return LOGGER
