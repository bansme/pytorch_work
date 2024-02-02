# -*- coding: utf-8 -*-
import logging
import logging.handlers

from flask import has_request_context, request

LEVEL_DICT = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARN": logging.WARN,
              "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}


class RequestFormatter(logging.Formatter):
    def format(self, record):
        if has_request_context():
            record.url = request.url
            record.remote_addr = request.remote_addr
        else:
            record.url = None
            record.remote_addr = None
        return super().format(record)


def add_logger_handler(app):
    f_handler = logging.FileHandler(app.config["LOG_FILE"])
    f_handler.setLevel(logging.INFO)

    f_format = RequestFormatter(
        "[%(asctime)s] %(remote_addr)s requested %(url)s "
        "%(levelname)s in %(module)s: %(message)s"
    )

    f_handler.setFormatter(f_format)

    app.logger.addHandler(f_handler)
    app.logger.setLevel(app.config["LOG_LEVEL"])
    return app


# 创建log日志输出
def create_logger(log_file, log_level=logging.DEBUG):
    """
    创建日志
    :param log_file:
    :param log_level:
    :return:
    """
    logger = logging.getLogger("logger")  # 获取logger

    handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=3,
                                                   encoding='utf-8')  # 实例一个输出流管道
    handler.setLevel(log_level)
    fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt)  # 实例化formatter
    handler.setFormatter(formatter)  # 为handler添加formatter
    logger.addHandler(handler)  # 为logger添加handler
    return logger

def get_logger(log_name):
    logger = logging.getLogger('logger')
    return logger
