# -*- coding: utf-8 -*-
"""
===============================================
PyFlask module
===============================================

========== ====================================
========== ====================================
 Module     PyFlask  module
 Date       2018-03-27
 Author     heewinkim
 Comment    `관련문서링크 <>`_
========== ====================================

*Abstract*
    * api 이름을 파라미터로 받는 Flask 클래스의 인스턴스를 Flask application 객체처럼 사용하시면 됩니다.
    * request 데이터의 경우 원래의 값 그대로 Flask의 args,files,form,json의 속성으로 저장됩니다.
    * input,output,elapse time 의 로깅처리가 사전구현 되어있습니다.

    * 인스턴스 마다 logging,output 객체가 생성됩니다. logging, error, output 의 경우 자체구현되어있어 따로 관련기능을 구현할 필요가 없습니다.
    * url 메소드를 추가하고 dict 형식의 결과만을 반환하면 됩니다.

    * helth check url : /<api_name>

    >>> EXAMPLE

    api_name = 'api'
    api_app = PyFlask(api_name)

    @api_app.route('/test/<int:param1>',methods=['POST','GET'])
    def request_test(param1):

        output = 'param = {}, form-data = {}'.format(param1,api_app.form)
        return output

    >>> 실제 구현 예시
    api_name = 'api'
    api_app = PyFlask(api_name)
    # api_instance = Api()

    # @api_app.route('/prefix/version/api_name',methods=['POST','GET'])
    # def request_test():
    #
    #     data = api_app.form['p']
    #     result = api_instance.run(data)
    #     return self.output.return_output(result)


    fd_app.application.run(host='0.0.0.0',port=5000)

    # $ curl -X POST --form 'p=test.jpg' http://0.0.0.0:5000/test/1234
    # $ curl -X POST --form 'p=test.jpg' http://0.0.0.0:5000/prefix/version/api_name/1234

===============================================
"""

import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
projroot_dir = os.path.dirname(parent_dir)
sys.path.insert(0,parent_dir)
sys.path.insert(0,projroot_dir)

import time
import json
from utilpack.core import *
from utilpack.core import PyLogger
from utilpack.core import PyOutput
from flask import Flask,request,jsonify
import traceback


class PyFlask(Flask):

    def __init__(self,api_name,useFileHandler=False,td_log=False):
        """
        url 경로를 위한 prefix,version,api_name을 인수로 받습니다.

        :param prefix_: prefix
        :param version_: version
        :param api_name_: api_name
        """
        super().__init__(api_name)
        self._logger = PyLogger(api_name,useFileHandler,td_log)
        self._api_name = api_name
        self.output = PyOutput()
        self.request = request
        self.is_health = False
        self._before_time = time.time()
        self.__add_urls()
        self.args = None
        self.form = None
        self.files = None
        self.json = None

    def __update_request_data(self):
        self.args = request.args
        self.form = request.form
        self.files = request.files
        self.json = request.json

    def __add_urls(self):

        self.after_request(self._after_request)
        self.before_request(self._before_request)
        self.url_value_preprocessor(self._preprocessing_request)
        self.register_error_handler(Exception,self._error_handler)
        self.add_url_rule('/',view_func=self._home, methods=['POST', 'GET'])
        self.add_url_rule('/{}'.format(self._api_name),view_func=self._health_check, methods=['POST', 'GET'])

    def _before_request(self):
        """
        요청 전에 실행됩니다.
        elapse time 저장변수 및 헬스체크에 대한 스위치 변수를 셋합니다.

        :return:
        """

        self._before_time = time.time()
        self.is_health = False

    def _after_request(self,response):
        """
        요청 후에 실행됩니다.
        elapse time을 로깅합니다.

        :param response:
        :return:
        """

        try:
            if self.is_health is False:
                # 요청부터 응답까지의 시간
                elapse = round((time.time() - self._before_time) * 1000,4)
                self._logger.info("output\t{}".format(list(response.response)[0].decode('utf-8')),elapse=elapse)

        except Exception:
            self._logger.warning(PyError(ERROR_TYPES.RUNTIME_ERROR,'logging time elapse failed - _after_request in PyFlask'))

        return response

    def _preprocessing_request(self,endpoint, values):
        """
        요청을 받은 후 처리직전에 실행됩니다.
        각 인풋데이터에 대한 처리를 진행합니다.

        :param endpoint:
        :param values:
        :return:
        """

        data = {}

        if request.form:
            data.update(request.form.to_dict())
        if request.args:
            data.update(request.args.to_dict())
        if request.files:
            for k,v in dict(request.files).items():
                data.update({k: v[0].filename})
        if request.json:
            data.update(dict(request.json))

        # 요청 처리전 요청된 클라이언트 정보를 입력합니다.
        self._logger.set_request_info(userIp=request.remote_addr,
                                      user_agent=request.headers.get('User-Agent'),
                                      svr_protocol=request.environ.get('SERVER_PROTOCOL'),
                                      req_method=request.method,
                                      path=request.path,
                                      full_path=request.full_path,
                                      host=request.host,
                                      language=data.get('language'),
                                      deviceId=data.get('deviceId'),
                                      appType=data.get('appType'),
                                      userNo=data.get('userNo')
                                      )

        if endpoint != '_health_check' and endpoint != '_home':
            self._logger.info('input\t{}'.format(json.dumps(data)))

        self.__update_request_data()

        return None

    def _error_handler(self,e):

        self._logger.error(traceback.format_exc())
        self.output.set_error(e)
        return self.output.get_output()

    def _health_check(self):
        """
        헬스 체크 URL 입니다.
        :return:
        """

        self.is_health = True
        return jsonify({'statusCode': 200, 'message': 'OK'})

    def _home(self):
        """
        루트 URL 입니다.
        :return:
        """
        self.is_health = True

        return 'Hello !'

    def validate_keys(self,object:dict,keys:list,value_check=False):
        """
        object 가장 최상단에서 주어진 Keys 들이 있는지를 체크합니다.
        key가 있는 경우 유효하나 키값인지(None) 체크합니다.


        :raise PARAMETER_ERROR:
        :param object: dictionary that json format
        :param keys: key list which have to include in object
        """

        for key in keys:
            if not key in object:
                raise PyError(ERROR_TYPES.PARAMETER_ERROR,'No {} data offered'.format(key))
            elif object.get(key) is None:
                if value_check:
                    raise PyError(ERROR_TYPES.PARAMETER_ERROR, 'Invalid {}'.format(key))


