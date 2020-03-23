# -*- coding: utf-8 -*-
"""
===============================================
PyError module
===============================================

========== ====================================
========== ====================================
 Module     PyError module
 Date       2019-03-26
 Author     heewinkim
========== ====================================

*Abstract*
    * 상위모듈에 전달하여 Main 또는 공통적으로 처리하는 방식을 지향합니다.
    * raise PyError(err_type,err_message) 형식으로 에러를 발생시킵니다.
    * 로깅시 PyError()로 에러인스턴스를 생성하여 전달합니다 eg. logger.error(PyError(err_type,err_message))
    * output.py 모듈의 set_error 사용시 인수로 발생한 에러를 그대로 전달합니다

        >>> EXAMPLE

        def do_something():
            raise PyError(ERROR_TYPES.RUPNTIME_ERROR,'test error')

        output = PyOutput()

        try:
            do something()
        except Exception as e:
            output.set_error(e)



===============================================
"""

import enum


class ERROR_TYPES(enum.Enum):
    """
    ERROR_TYPES 클래스
    enum타입으로 에러 종류가 선언 되어 있는 클래스 입니다.

    에러 추가 방식은 ERROR_TYPES에서 에러코드에 맞게 새로운 에러멤머를 추가하신후
    형식에 맞게(주석 참조) 에러클래스를 작성해주시면 됩니다.
    """

    # 400 ERROR
    REQUEST_ERROR = 0x40000
    PARAMETER_ERROR = 0x40001

    # 404 ERROR
    RESOURCE_ERROR = 0x40400

    # 500 ERROR
    UNEXPECTED_ERROR = 0x50000

    # 600 PY CUSTOM ERROR
    IMAGE_READ_ERROR = 0x60000
    IMAGE_SIZE_ERROR = 0x60001
    IMAGE_FORMAT_ERROR = 0x60002
    INPUT_KEY_ERROR = 0x60003
    INPUT_VALUE_ERROR = 0x60004

    # 700 PY PRIVATE ERROR
    INITIATE_ERROR=0x70000
    PREPROCESSING_ERROR = 0x70001
    IMAGE_PROCESS_ERROR = 0x70002
    RUNTIME_ERROR=0x70003
    MODEL_RUNTIME_ERROR = 0x70004
    POSTPROCESSING_ERROR = 0x70005


class PyError(Exception):

    def __init__(self,err_type,message=''):
        """
        example : PyError(ERROR_TYPES.IMAGE_READ_ERROR,'failed to download the image')

        Autor : heewinkim

        :param err_type: 에러타입(ERROR_TYPES 멤버)
        :param message: 상세 에러 내용
        """

        self.err_type = err_type
        self._err_message = message

    def __str__(self):
        if self._err_message:
            return '[{}] {}'.format(self.err_type.name.upper(),self._err_message)
        else:
            return '[{}] {}'.format(self.err_type.name.upper(),self.err_type.name.replace('_',' '))

