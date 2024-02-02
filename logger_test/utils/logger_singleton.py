# -*- coding: utf-8 -*-
import logging


def singleton(cls):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton


@singleton
class Logger():

    def __init__(self, log_file, log_level=logging.DEBUG):
        self.log_file = log_file
        self.log_level = log_level
        self.logger = logging.getLogger("logger")  # 获取logger
        self.set_logger()


    def set_logger(self):

        handler = logging.handlers.RotatingFileHandler(self.log_file, maxBytes=1024 * 1024, backupCount=3,
                                                       encoding='utf-8')  # 实例一个输出流管道
        handler.setLevel(self.log_level)
        fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        formatter = logging.Formatter(fmt)  # 实例化formatter
        handler.setFormatter(formatter)  # 为handler添加formatter
        self.logger.addHandler(handler)  # 为logger添加handler

