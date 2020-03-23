# -*- coding: utf-8 -*-
"""
===============================================
data module
===============================================

========== ====================================
========== ====================================
 Module     data module
 Date       2019-10-10
 Author     heewinkim
 Comment    `관련문서링크 <>`_
========== ====================================

*Abstract*

    * zip 파일,데이터에 대한 처리기능을 제공합니다.

===============================================
"""


import io
import zipfile
import json
import pickle
from .error import *


class Zip(object):

    @staticmethod
    def decompress(data, filename=None):
        """
        decompress -> bytes array

        :param data : zip filepath or zip format bytes array
        :param filename: filename in zip if None, extractall
        """
        try:
            if type(data)==str:
                with open(data, 'rb') as f:
                    zip_f = zipfile.ZipFile(f)
                    if filename:
                        return zip_f.read(filename)
                    else:
                        return [zip_f.read(filename) for filename in zip_f.namelist()]
            elif type(data)==bytes:
                zip_f = zipfile.ZipFile(io.BytesIO(data))
                bytes_data = zip_f.read(filename)
                return bytes_data
        except Exception:
            raise PyError(ERROR_TYPES.PARAMETER_ERROR,'Invalid metaFile, failed to decompress zipfile')

    @staticmethod
    def compress(data, filename,encode='utf-8'):
        """
        compress -> byteszz

        :param data : zip filepath or zip format bytes array
        :param filename: filename which will be compress
        :param encode: encode charset , if data is string type
        :return: bytes array
        """
        try:
            if type(data)==str:
                data = data.encode(encode)
            elif type(data) == bytes:
                pass

            in_memory = io.BytesIO()

            zf = zipfile.ZipFile(in_memory, mode="w")
            zf.writestr(filename, data)
            zf.close()

            in_memory.seek(0)

            return in_memory.read()
        except Exception:
            raise PyError(ERROR_TYPES.PREPROCESSING_ERROR,'Failed to compress zipfile')



class PyData(object):

    zip=Zip

    @staticmethod
    def save_json(data,path):
        with open(path,'w') as f:
            f.write(json.dumps(data))

    @staticmethod
    def load_json(data,fromfile=False):
        if fromfile:
            return json.loads(open(data).read())
        else:
            return json.loads(data)

    @staticmethod
    def save_pickle(data,path):
        """
        데이터를 바이너리파일로 저장합니다.

        :param path: 저장 경로
        :param data: 저장할 데이터
        :return: None
        """

        with open(path,'wb') as f:
            pickle.dump(data,f)

    @staticmethod
    def load_pickle(path):
        """

        :param path: 저장 경로
        :return: None
        """

        with open(path,'rb') as f:
            data = pickle.load(f)
            return data