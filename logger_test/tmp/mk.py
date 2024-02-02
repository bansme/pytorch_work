# -*- coding: utf-8 -*-
import logging
from utils.logger_singleton import Logger

def mk_test():

    logger = logging.getLogger("logger.mk")

    logger.warning("mk test")


def mk_test_single():
    logger = Logger("").logger
    logger.warning("2222")
