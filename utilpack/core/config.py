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

    def __init__(self,config_dirpath=current_dir):
        """
        common 파일 로드 및 데이터 저장

        :param config_dirpath: py_api.conf 파일 경로
        """
        configFilePath = config_dirpath + "/py_api.conf"

        self.config = configparser.ConfigParser()
        self.config.read(configFilePath, encoding='utf-8')

        self.log_path = self.config["LOG_INFO"]["LOG_PATH"]
        self.log_rotate = bool(self.config["LOG_INFO"]["LOG_ROTATE"]=='True')

        self.td_ip = self.config["LOG_INFO"]["TD_IP"]
        self.td_port = self.config["LOG_INFO"]["TD_PORT"]
        self.td_tag = self.config["LOG_INFO"]["TD_TAG"]
