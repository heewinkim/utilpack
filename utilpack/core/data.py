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
from pdfrw import PdfReader, PdfWriter, PdfDict, PdfName
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


class Pdf(object):

    @staticmethod
    def add_metadata(pdf_path, save_path, dict_data, keyname='metadata'):
        """
        add metadata(dict) in pdf

        :param pdf_path: pdf file path eg. example/path/file.pdf
        :param save_path: path to save eg. path/to/save/file.pdf
        :param dict_data: python dictionary data , allow any depth or any data type include
        :param keyname: feild name in pdf metadata, default=metadata
        :return: None
        """
        metadata = PdfDict()
        metadata[PdfName(keyname)] = json.dumps(dict_data)
        trailer = PdfReader(pdf_path)
        trailer.Info.update(metadata)
        PdfWriter().write(save_path, trailer)

    @staticmethod
    def read_metadata(pdf_path, keyname='metadata'):
        """
        read metadata(dict) which in pdf

        :param pdf_path: pdf file path eg. example/path/file.pdf
        :param keyname: feild name in pdf metadata, default=metadata
        :return: dict, metadata
        """
        trailer = PdfReader(pdf_path)
        metadata = json.loads(trailer.Info['/{}'.format(keyname)][1:-1])
        return metadata


class PyData(object):

    zip = Zip
    pdf = Pdf

    @staticmethod
    def save_json(data,path):
        with open(path,'w') as f:
            f.write(json.dumps(data))

    @staticmethod
    def load_json(data,fromfile=True):
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


if __name__ == '__main__':

    # 데이터를 압축합니다.
    compress_data = PyData.zip.compress('string data',filename='filename')
    print(compress_data)  # b'PK\x03\x04\x14...

    # 압축된 데이터를 읽습니다.
    decompress = data = PyData.zip.decompress(compress_data,filename='filename')
    print(decompress)  # b'string data'

    # 데이터를 json 형태로 저장합니다.
    PyData.save_json({'a':1,'b':2},'sample.json')

    # json 데이터파일을 읽습니다.
    data = PyData.load_json('sample.json')
    print(data)  # {'a': 1, 'b': 2}

    # 데이터를 pickle 형태로 저장합니다.
    PyData.save_pickle({'a':1,'b':2},'sample.pkl')

    # pickle 데이터파일을 읽습니다.
    data = PyData.load_pickle('sample.pkl')
    print(data)  # {'a': 1, 'b': 2}

    # pdf 파일에 메타데이터를 추가합니다.
    PyData.pdf.add_metadata('file.pdf','save.pdf',{'a':1,'b':2,'c':3},'keyname')

    # pdf 파일에 있는 메타데이터를 읽어옵니다.
    metadata = PyData.pdf.read_metadata('file.pdf','keyname')
    print(metadata)  # {'a': 1, 'b': 2, 'c': 3}