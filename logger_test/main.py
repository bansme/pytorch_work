import os
from utils.logger import create_logger
from utils.logger_singleton import Logger
from tmp.mk import mk_test, mk_test_single
from tmp2.mkk import mkk_test
import configparser


def load_conf_info(conf_path, section_name=None):
    """
    默认返回所有section的配置字段
    :param conf_path: 配置文件路径
    :param section_name: 解析指定section
    :return: 以字典形式返回数据
    """
    res = {}
    try:
        conf = configparser.RawConfigParser()
        conf.read(conf_path, encoding='utf-8')
        if section_name is None:
            sections = conf.sections()
            for key in sections:
                res[key] = dict(conf.items(key))
        else:
            res[section_name] = dict(conf.items(section_name))
    except Exception as exp:
        print(str(exp))
        res = None
    return res


def main():

    logger = create_logger("logs/test.log", )
    logger.warning("main process")
    mk_test()
    mkk_test()
    #
    # logger = Logger("logs/test.log").logger
    # logger.warning("singleton")
    #
    # mk_test_single()



if __name__ == '__main__':
    main()

