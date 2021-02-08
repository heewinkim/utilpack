# -*- coding: utf-8 -*-
"""
===============================================
config module
===============================================

========== ====================================
========== ====================================
 Module     config module
 Date       2019-03-26
 Author     heewinkim
========== ====================================

*Abstract*

    * config 읽기 모듈입니다.

        >>> EXAMPLE
        py_config = PyConfig()
        print(py_config.log_path)

===============================================
"""

import os
import configparser

current_dir = os.path.dirname(os.path.abspath(__file__))


class PyConfig(object):
    """
    PyConfig 클래스

    스냅스 API 운영시 configuration에 대한
    처리를 담당하는 클래스 입니다.

    """

    def __new__(cls,config_dirpath=current_dir):
        configFilePath = config_dirpath + "/py.conf"
        config = configparser.ConfigParser()
        config.read(configFilePath, encoding='utf-8')
        return config


if __name__ == '__main__':

    py_config = PyConfig()
    print(py_config['LOG']['PATH'])