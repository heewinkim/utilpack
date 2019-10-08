# -*- coding: utf-8 -*-
"""
===============================================
logger module
===============================================

========== ====================================
========== ====================================
 Module     logger module
 Date       2019-03-26
 Author     hian
========== ====================================

*Abstract*
    * 설정값을 로드하여 제공하는 모듈입니다.
    * 로그 설정에 대한 처리(자체 로커 로테이션 사용 or 리눅스 logrotate 사용)
    * 외부에서 사용시 인스턴스화 된 py_logger 를 import 하여 사용합니다.
    * set_input2log 를 통해 요청마다 클라이언트 정보를 입력해주어야 합니다.

    >>> EXAMPLE
    import traceback

    logger = PyLogger()


    def do_something():
        raise ValueError

    try:
        do_something()

    except Exception as e:
        logger.logger.error(traceback.format_exc())

    * TODO info 로그에 error로그가 포함되는 문제 개선필요


===============================================
"""
import time, os
import logging
import requests
import json
from datetime import datetime
from shutil import copyfile
from .config import PyConfig
from .singleton import Singleton
monthes = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
           'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
           'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12',
           }

py_config = PyConfig()


class PyLogger(logging.Filter,metaclass=Singleton):

    def __init__(self, log_name,td_log=False):

        """
        init function

        :param log_name: log 파일에 접미사로 붙을 이름
        :param log_dir: log파일이 저장될 경로 default = ./log , config 설정 존재시 설정값 적용
        """

        # fluentd-log parameters
        self.td_log=td_log
        self.td_ip = py_config.td_ip
        self.td_port = py_config.td_port
        self.td_tag = py_config.td_tag

        # 오늘날짜 시간 저장
        self.saveDate = self._get_today()

        # 로그 저장 경로
        self.log_dir = py_config.log_path +'/' + log_name
        self.log_name = log_name

        if self.td_log:
            self.logs_type = ["error"]
        else:
            self.logs_type = ["info", "error"]

        self.logs_info = {"format":".log"}

        self.__logger = logging.getLogger(self.log_name)
        self.__logger.setLevel(logging.INFO)

        # 포매터를 만든다
        self.formatter = logging.Formatter(
            '%(levelname)s\t%(process)d\t%(asctime)s\t%(user_ip)s\t%(user_agent)s\t%(svr_protocol)s\t%(req_method)s\t%(full_path)s\t%(message)s',
            datefmt='%y-%m-%d %H:%M:%S')

        # input image path 설정
        super().__init__()
        self.__logger.addFilter(self)
        self.set_request_info()
        self._set_all_logger()

        if py_config.log_rotate:
            self._check_init_copiedlog()

    def info(self,message,**kwargs):
        """
        info log를 남깁니다.
        PyLogger 초기화시 td_log 가 True이면
        kwargs를 추가하여 td-log를 전송합니다.

        :param message: 메세지
        :param kwargs: td log에 추가할 키워드 파라미터
        :return:
        """
        if self.td_log:
            data = {
                'language':self.accept_language,
                'deviceId':self.deviceId,
                'appType':self.appType,
                'userIp':self.userIp,
                'processId':os.getpid(),
                'userNo':self.userNo,
                'user-agent':self.user_agent,
                'type':'REQ' if message.split('\t')[0]=='input' else 'RES',
                'hostname':self.host,
                'uri':self.path,
                'path_var':self.full_path,
                'method': self.req_method,
                'event_time': datetime.now().strftime('%Y%m%d%H%M%S.%f')[:-3],
                'payload': json.loads(message.split('\t')[1]),
            }
            for k,v in kwargs.items():
                data[k]=v

            requests.post('http://{}:{}/{}'.format(self.td_ip,self.td_port,self.td_tag),json=data)
        else:
            for k,v in kwargs.items():
                self.__logger.info('{}\t{}'.format(k,v))
            self.__logger.info(message)

    def warning(self,message):
        self.__logger.warning(message)

    def error(self,message):
        self.__logger.error(message)

    def set_request_info(self,
                         userIp=None,
                         user_agent=None,
                         svr_protocol=None,
                         req_method=None,
                         path=None,
                         full_path=None,
                         host=None,
                         language=None,
                         deviceId=None,
                         appType=None,
                         userNo=None
                         ):
        """
        input으로 들어온 데이터를 로그에 기록합니다.

        :param user_ip: user ip
        :param user_agent: user agent
        :param svr_protocol:  server protocol
        :param req_method: GET or POST
        :param full_path: url full_path
        :return:
        """

        self.userIp = userIp
        self.user_agent = user_agent
        self.svr_protocol=svr_protocol
        self.req_method=req_method
        self.path=path
        self.full_path=full_path
        self.host=host
        self.accept_language=language
        self.deviceId=deviceId
        self.appType=appType
        self.userNo=userNo

    def filter(self, record):
        """
        오버로딩 메소드
        로그에 input 이미지 경로를 나타내는 필터

        Autor : hian

        :param record: input data
        :return:

        """

        record.user_ip = self.userIp
        record.user_agent = self.user_agent
        record.svr_protocol = self.svr_protocol
        record.req_method = self.req_method
        record.full_path = self.full_path
        return True

    def _set_all_logger(self) -> None:
        """
        모든 로그 모듈 설정

        """
        for log_type in self.logs_type:

            self._set_one_logger(log_type=str(log_type))

    def _set_one_logger(self, log_type) -> None:
        """
        한개 로그 모듈 설정

        :param log_type: (str) 'info' or 'error'
        """

        log_save_dir = "{}/{}_log/".format(self.log_dir, log_type)
        log_save_name = "{}_{}".format(self.log_name, log_type)

        self.logs_info[log_type] = log_save_name
        self.logs_info[log_type+"_dir"] = log_save_dir

        # 로그 저장 경로 존재 확인 및 생성
        try:
            if not os.path.exists(log_save_dir):
                os.makedirs(log_save_dir)
                print("The log dir was created: ", log_save_dir)
                time.sleep(0.3)
        except FileExistsError as e:
            print("log dir exist, ignore create command..")
            time.sleep(0.1)

        # 스트림과 파일로 로그를 출력하는 핸들러를 각각 만든다.
        fileHandler = logging.FileHandler(log_save_dir+log_save_name + self.logs_info["format"])

        # 각 핸들러에 포매터를 지정한다.
        fileHandler.setFormatter(self.formatter)

        # 로그 레벨 설정
        if log_type == "info":
            fileHandler.setLevel(logging.INFO)

        elif log_type == "error":
            fileHandler.setLevel(logging.ERROR)

        # 로거 인스턴스에 스트림 핸들러와 파일핸들러를 붙인다.
        self.__logger.addHandler(fileHandler)

    def _check_init_copiedlog(self) -> None:
        """
        재 실행시, 로그 복사본 있는지 확인

        """
        for log_type in self.logs_type:

            ori_log_path = "{}{}{}".format(self.logs_info[log_type + "_dir"],
                                           self.logs_info[log_type],
                                           self.logs_info["format"])

            preDate = None

            try:
                f = open(ori_log_path, 'r')

                preDate = f.readlines()

                f.close()

            except Exception as e:
                print('No Log File : write logs on {}{}{}'.format(self.logs_info[log_type+"_dir"],self.logs_info[log_type],self.logs_info['format']))

            if len(preDate)>0:

                firstLog = preDate[0]

                # 날짜 단위
                firstDate = firstLog.split(" ")[1].split(' ')[0]

                if firstDate != self.saveDate:
                    copy_log_path = "{}{}_{}{}".format(self.logs_info[log_type + "_dir"],
                                                       self.logs_info[log_type],
                                                       firstDate,
                                                       self.logs_info["format"])

                    self._copy_clean_log(copy_log_path, ori_log_path)

    def _copy_clean_log(self,copy_log_path, ori_log_path) -> None:
        """
        로그 파일 복사 및 원본 로그 내 기록 지우기

        :param copy_log_path: 
        :param ori_log_path: 
        """
        # copy가 존재하지 않으면
        if os.path.exists(copy_log_path) is False:

            # 로그파일의 복사본을 생성 한다.
            copyfile(ori_log_path, copy_log_path)

            # check again
            if os.path.exists(copy_log_path):
                # 기존 로그 파일의 데이터는 지운다. (파일 자체는 유지)
                f = open(ori_log_path, 'w')
                f.close()

        else:
            print("already exist log path: ",copy_log_path)

    def _make_new_log(self) -> None:
        """
        모든 로그에 대한 로그복사본 검사 및 없으면 생성

        """
        for log_type in self.logs_type:

            # log 복사 시간
            log_copy_date = self._get_today()

            # 복사 할 log 파일 경로
            copy_log_path = "{}{}_{}{}".format(self.logs_info[log_type + "_dir"],
                                               self.logs_info[log_type],
                                               log_copy_date,
                                               self.logs_info["format"])

            ori_log_path = "{}{}{}".format(self.logs_info[log_type + "_dir"],
                                           self.logs_info[log_type],
                                           self.logs_info["format"])

            try:
                self._copy_clean_log(copy_log_path, ori_log_path)

            except Exception as e:
                self.__logger.error("error: ", e)

    def check_log_date(self):
        """
        현재 날짜와 저장된 날짜 다른지 확인
        """

        if bool(py_config.log_rotate) is True:

            nowdate = self._get_today()

            # log 파일 복사
            if self.saveDate != nowdate:
                self._make_new_log()
                self.saveDate = nowdate

    @staticmethod
    def _get_today():
        """
        YY-MM-DD 형식의 오늘날짜를 str으로 반환(eg. '18-07-08')
        :return: str, 오늘날짜
        """

        ctime_list = time.ctime().split(' ')
        today = ctime_list[-1][2:]+'-'+monthes[ctime_list[1]]+'-'+'%02d' % int(ctime_list[-3])

        return today
