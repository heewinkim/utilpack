# -*- coding: utf-8 -*-
"""
===============================================
output module
===============================================

========== ====================================
========== ====================================
 Module     output module
 Date       2019-03-26
 Author     heewinkim
 Comment    `관련문서링크 <>`_
========== ====================================

*Abstract*
    * data output 처리를 담당합니다.
    * 최종 아웃풋을 json 형태로 제공
    * set_error(e) 메소드 제공


    output = PyOutput()

    output.set_output(a)
    output.set_success()

    print(output.get_output())
    # {"statusCode": 200, "message": "success", "a": 0, "b": [1, 2, 3]}

    print(output.get_output())
    # {"statusCode": None, "message": None, "a": None, "b": None}

===============================================
"""

import json
import numpy as np
from .error import *
from .singleton import Singleton


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class PyOutput(object,metaclass=Singleton):

    def __init__(self,top_keys=['statusCode','message']):

        self.output={k: None for k in top_keys}
        self.default_obj=None
        self.reset()

    def reset(self)-> None:
        """
        최상단의 object 값들을 None로 리셋합니다.
        :return:
        """

        if self.default_obj:
            self.output = self.default_obj.copy()
        elif self.output:
            self.output = self.output.fromkeys(self.output,None)
        else:
            self.output = {k: None for k in ['statusCode','message']}

    def set_default(self,**kwargs):
        """
        keys,default_value를 받아 키를 하나의 값으로 초기화 하거나
        dict_obj 값을 받아 output 디폴트 값을 update 합니다


        keys: 디폴트로 지정하고자 하는 key 리스트(default_value로 초기화 됨)

        default_value: keys를 넘겼을때 초기화되는 값

        dict_obj: dictionary 형태로 값을 받아 디폴드 값으로 지정

        :return: None
        """

        if 'keys' in kwargs:
            keys = kwargs.get('keys')
            default_value = kwargs.get('default_value',None)
            self.default_obj = {k: default_value for k in keys}

        elif 'dict_obj' in kwargs:
            self.default_obj = kwargs.get('dict_obj')

        else:
            raise PyError(ERROR_TYPES.PREPROCESSING_ERROR,'Invalid parameter offered - set_default in PyOutput')

        self.output = self.default_obj.copy()

    def set_output(self,output:dict) -> None:
        """
        dictionary 형태의 아웃풋을 저장합니다.
        python dict.update와 동일한 기능을 제공합니다.
        기존값을 업데이트 또는 새롤운 키-값 쌍을 추가합니다.

        :param output: dictionary
        :return: None
        """
        self.output.update(output)

    def get_output(self):
        """
        output값을 반환 및 리셋합니다.

        :return: json format data
        """

        output = json.dumps(self.output,cls=NumpyEncoder)
        self.reset()
        return output

    def set_error(self,e):
        """
        에러에 대한 후처리를 합니다.

        :param e: error type
        """

        if hasattr(e,'err_type'):
            if e.err_type.name in ERROR_TYPES.__members__ and e.err_type.value in ERROR_TYPES._value2member_map_:

                statusCode = e.err_type.value >> 16

                # 400 ERROR
                if statusCode == 4:
                    self.output['statusCode'] = 400
                    self.output['message'] = str(e)

                # 500 ERROR
                elif statusCode == 5:
                    self.output['statusCode'] = 500
                    self.output['message'] = str(e)

                # 600 ERROR
                elif statusCode == 6:
                    self.output['statusCode'] = 200
                    self.output['message'] = str(e)

                # 700 ERROR
                elif statusCode == 7:
                    self.output['statusCode'] = 500
                    self.output['message'] = '[RUNTIME_ERROR] Unexpected Error Occurred.'
            else:
                self.output['statusCode'] = 500
                self.output['message'] = '[RUNTIME_ERROR] Unexpected Error Occurred.'
        else:
            if hasattr(e, 'code') and e.code == 400:
                self.output['statusCode'] = 400
                self.output['message'] = '[REQUEST_ERROR] Bad Request.'
            elif hasattr(e, 'code') and e.code == 401:
                self.output['statusCode'] = 401
                self.output['message'] = '[REQUEST_ERROR] Method not allowed.'
            elif hasattr(e, 'code') and e.code == 402:
                self.output['statusCode'] = 402
                self.output['message'] = '[REQUEST_ERROR] Unauthorized request.'
            elif hasattr(e, 'code') and e.code == 403:
                self.output['statusCode'] = 403
                self.output['message'] = '[REQUEST_ERROR] Forbidden Resource.'
            elif hasattr(e, 'code') and e.code == 404:
                self.output['statusCode'] = 404
                self.output['message'] = '[RESOURCE_ERROR] Invalid Resource requested.'
            elif hasattr(e, 'code') and e.code == 405:
                self.output['statusCode'] = 405
                self.output['message'] = '[REQUEST_ERROR] Method not allowed.'
            else:
                self.output['statusCode'] = 500
                self.output['message'] = '[RUNTIME_ERROR] Unexpected Error Occurred.'

    def set_success(self, status=200, message='success'):
        """
        성공처리

        :param status: 상태코드
        :param message: 메세지
        """
        self.output['statusCode'] = status
        self.output['message'] = message

    def return_output(self,result):
        self.set_output(result)
        self.set_success()
        return self.get_output()


if __name__ == '__main__':

    # PyOutput는 singletone 디자인패턴으로 객체화 합니다.
    # PyOutput은 PyFlask에서 사용되어 REST API 구현시 json output을 관리합니다.
    output = PyOutput(top_keys=['statusCode','message'])

    # 저장된 output.output 데이터를 가져오며, 자동적으로 저장된 output.output 데이터는 초기화 됩니다.
    # 스택의 pop과 비슷하다고 보면 됩니다.
    print(output.get_output())  # {"statusCode": null, "message": null}

    # keys,default_value를 받아 키를 하나의 값으로 초기화 하거나, dict_obj 값을 받아 output 디폴트 값을 update 합니다
    # PyOutput 객체를 새로 생성하지 않고 초기값을 변경하는 것과 같습니다.
    output.set_default(dict_obj={'example':'value'})
    print(output.output)  # {'example': 'value'}

    # output 데이터를 할당합니다.
    output.set_output({'data':1234})
    print(output.output)  # {'example': 'value', 'data': 1234}

    # 현재 API의 진행상황에 에러가 있음을 표시합니다. (statusCode, message 데이터로 표기)
    # PyError와 호환됩니다.
    output.set_error(PyError(ERROR_TYPES.IMAGE_READ_ERROR,'example error'))
    print(output.output)  # {'example': 'value', 'data': 1234, 'statusCode': 200, 'message': '[IMAGE_READ_ERROR] example error'}

    # 현재 API의 진행상황이 성공적임을 표시합니다. (statusCode, message 데이터로 표기)
    output.set_success()

    # set_output(data) -> set_success() -> reset() 의 일련과정을 한번에 처리하는 strategy 패턴과 같은 메서드이며 output 데이터를 반환합니다.
    returned_data = output.return_output({'example':'data'})
    print(returned_data)  # {"example": "data", "data": 1234, "statusCode": 200, "message": "success"}

    # output.output인 데이터를 초기화 합니다.
    output.reset()

