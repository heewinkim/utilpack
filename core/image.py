# -*- coding: utf-8 -*-
"""
===============================================
image module
===============================================

========== ====================================
========== ====================================
 Module     image module
 Date       2019-03-26
 Author     heewinkim
========== ====================================

*Abstract*
    * PyImage 클래스를 제공합니다.
    * 이미지 관련한 처리 함수를 제공합니다.

===============================================
"""

from .error import *

import cv2
import urllib.request
import numpy as np
from PIL import Image
from io import BytesIO
import base64


class PyImage(object):

    @staticmethod
    def calculate_sizerate(ot, w, h):
        """
        사이즈 비율 계산

        :param ot: 회전각도 (0~8)
        :param w: 사진 가로 길이 (1~32767)
        :param h: 사진 세로 길이 (1~32767)
        :return: 계산 된 사이즈 비율
        """"""
        """
        whRange = [0, 32768]
        size_r = 0

        if (whRange[1] > w > whRange[0]) and (whRange[1] > h > whRange[0]):

            if 0 < ot <= 4:
                size_r = w / h

            elif 9 > ot > 4:
                size_r = h / w

            elif ot == 0:
                size_r = 1

        return size_r

    @staticmethod
    def bytes2cv(img_data):
        """
        byte 데이터를 cv 이미지로 변환합니다.
        에러 발생시 PyError.EncodeByte2CvError를 발생시킵니다.

        :param img_data: byte image data
        :return: cv 이미지
        :raise IMAGE_PROCESS_ERROR:
        """

        img_cv = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        if img_cv is None:
            raise PyError(ERROR_TYPES.IMAGE_PROCESS_ERROR,'failed to bytes2cv image in pyImage')

        return img_cv

    @staticmethod
    def cv2bytes(img_cv):
        return cv2.imencode('.jpg', img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tostring()

    @staticmethod
    def read_bytedata(data, img_type):
        """
        입력타입 ['url','filepath','bytes']에 따라 data를 byte image data로 로드합니다.


        :param data: input data
        :param img_type: input img_type , ['url','filepath','bytes','b64'] 중 하나
        :return: byte image data
        :raise IMAGE_READ_ERROR:
        """


        try:
            bytes_data = None

            # 각 파일타입에 따른 데이터 읽기
            if img_type == 'bytes':
                bytes_data = data.read()
            elif img_type == 'url':
                if data.startswith('/Upload'):
                    data = 'https://www.py.com' + data
                response = urllib.request.urlopen(data)
                if response.getcode() == 200:
                    bytes_data = response.read()
            elif img_type == 'filepath':
                with open(data, 'rb') as f:
                    bytes_data = f.read()
            elif img_type == 'b64':
                bytes_data = base64.b64decode(data)

            return bytes_data

        except Exception:
            raise PyError(ERROR_TYPES.IMAGE_READ_ERROR,'failed to read image - read_bytedata in PyImage')

    @staticmethod
    def rotate_image(img, orientation: int = 0,copy=False):
        """
        orientation 값을 참조하여 원본 이미지로 복구합니다.

        :param img: array,image
        :param orientation: int,0~8
        :param copy: 복사본을 반환, 결과를 이미지처리 할때 필요
        :return: array, rotated img
        """

        orientation = int(orientation)

        if orientation in [0, 1]:
            result_img =  img
        elif orientation == 2:
            result_img =  np.fliplr(img)
        elif orientation == 3:
            result_img =  np.rot90(img, 2)
        elif orientation == 4:
            result_img =  np.fliplr(np.rot90(img, 2))
        elif orientation == 5:
            result_img =  np.fliplr(np.rot90(img, 3))
        elif orientation == 6:
            result_img =  np.rot90(img, 3)
        elif orientation == 7:
            result_img =  np.fliplr(np.rot90(img, 1))
        elif orientation == 8:
            result_img =  np.rot90(img, 1)
        else:
            result_img = img

        if copy:
            return result_img.copy()
        else:
            return result_img

    @staticmethod
    def cv2base64(img_cv,tostring=False):
        """
        cv2 이미지를 base64_jpeg 포맷으로 변환합니다

        :param img_cv: array, bgr cv2 image
        :return: array, base64 image
        :raise IMAGE_PROCESS_ERROR:
        """

        encoded_img = cv2.imencode('.jpg', img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1]

        if tostring:
            img_b64 = base64.b64encode(encoded_img).decode('utf-8')
        else:
            img_b64 = np.array([base64.urlsafe_b64encode(encoded_img)])

        if img_b64 is None:
            raise PyError(ERROR_TYPES.IMAGE_PROCESS_ERROR,'failed to convert cv2 to base64 - cv2base64 in PyImage')
        else:
            return img_b64

    @staticmethod
    def byte2base64(bytes_data):
        """
        byte data 이미지를 base64_jpeg 포맷으로 변홥합니다.

        :param bytes_data: byte data
        :return: array, base64 image
        :raise IMAGE_PROCESS_ERROR:
        """

        img_b64 = np.array([base64.urlsafe_b64encode(bytes_data)])

        if img_b64 is None:
            raise PyError(ERROR_TYPES.IMAGE_PROCESS_ERROR,'failed to convert bytes to base64 - byte2base64 in PyImage')
        else:
            return img_b64

    @staticmethod
    def check_image_shape(img_cv) -> None:
        """
        입력 이미지 shape 확인. 드물게, 3차원 배열이 아닌 4차원 배열로 들어오는 이미지가 존재함.
        뎁쓰 값이 3이 아닐시에 PyError.ImageShapeError 를 발생시킵니다.

        :param img_cv: cv 이미지
        """
        if img_cv.shape[2] is not 3:
            raise PyError(ERROR_TYPES.IMAGE_FORMAT_ERROR, 'wrong image shape - check_image_shape in PyImage')

    @staticmethod
    def check_img_sz_fmt(bytes_data, min_size=(20, 20), max_size=(10000, 10000),
                         allowed_extensions={'png', 'jpg', 'jpeg', 'bmp'}) -> None:
        """
                이미지 헤더를 읽어 사이즈와 포맷을 체크 합니다.
        사이즈가 너무작거나 큰경우, 지원하지 않는 이미지 포맷인경우 에러를 발생시킵니다.

        :param bytes_data: byte image data
        :param min_size: tuple, 최소 사이즈 (width,height)
        :param max_size: tuple, 최대 사이즈 (width,height)
        :param allowed_extensions: 지원 가능 이미지 확장자
        :raise IMAGE_FORMAT_ERROR:
        """
        img = Image.open(BytesIO(bytes_data))

        w = img.width
        h = img.height
        fmt = img.format

        if (h < min_size[1] and w < min_size[0]) or (w * h < min_size[0] * min_size[1]):
            raise PyError(ERROR_TYPES.IMAGE_FORMAT_ERROR, 'image is too small in check_img_sz_fmt')
        if (h > max_size[1] and w > max_size[0]) or (w * h > max_size[0] * max_size[1]):
            raise PyError(ERROR_TYPES.IMAGE_FORMAT_ERROR, 'image is too big in check_img_sz_fmt')
        if not fmt.lower() in allowed_extensions:
            raise PyError(ERROR_TYPES.IMAGE_FORMAT_ERROR, 'not supported image format in check_img_sz_fmt')

    @staticmethod
    def preprocessing_image(data, img_type, img_ot, cvt_type='cv2'):
        """
        이미지 전처리를 수행합니다

        전체과정
        [cv2]
        input data -> byte image data -> image check ->
        convert to cv image -> check image shape -> rotate image
        [b64]
        input data -> byte image data -> image check ->
        if ot is 0 or 1, convert to b64 image
        else, convert to cv image -> check image shape -> rotate image -> convert to b64 image

        :param data: img_type에 따른 input data
        :param img_type: bytes, url, filepath 중 하나s
        :param img_ot: image orientation 값
        :param cvt_type: 'cv2', or 'b64' default : cv2
        :return: cvt_type의 이미지

        :raise IMAGE_READ_ERROR:
        :raise IMAGE_FORMAT_ERROR:
        :raise IMAGE_PROCESS_ERROR:
        """

        bytes_data = PyImage.read_bytedata(data, img_type)
        PyImage.check_img_sz_fmt(bytes_data)

        if cvt_type == 'cv2':

            img_cv = PyImage.bytes2cv(bytes_data)
            img_cv = PyImage.rotate_image(img_cv, orientation=img_ot)
            PyImage.check_image_shape(img_cv)
            return img_cv

        elif cvt_type == 'b64':

            if int(img_ot) not in [0, 1]:
                img_cv = PyImage.bytes2cv(bytes_data)
                img_cv = PyImage.rotate_image(img_cv, orientation=img_ot)
                PyImage.check_image_shape(img_cv)
                return PyImage.cv2base64(img_cv)
            else:
                return PyImage.byte2base64(bytes_data)
