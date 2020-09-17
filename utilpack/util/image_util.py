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
    * 이미지 관련 유틸 모음

===============================================
"""

import os
from utilpack.core import *

import cv2
import glob
import datetime
import numpy as np
import urllib.request
from math import ceil
from PIL import Image,ImageDraw,ImageFont
from PIL import PngImagePlugin,JpegImagePlugin,BmpImagePlugin
from io import BytesIO
import matplotlib.pyplot as plt
from PIL.ExifTags import TAGS, GPSTAGS
current_dir = os.path.dirname(os.path.abspath(__file__))


class PyImageUtil(object):

    cv2 = cv2
    plt = plt

    @staticmethod
    def images2pdf(img_list, save_path='./images.pdf', color_mode='bgr'):
        if color_mode == 'bgr':
            img_list = [Image.fromarray(img[..., ::-1]) for img in img_list]
        else:
            img_list = [Image.fromarray(img) for img in img_list]
        img_list[0].save(save_path, "PDF", resolution=100.0, save_all=True, append_images=img_list[1:])

    @staticmethod
    def putTextCenter(img, text, org, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1., color=(0, 0, 0),thickness=2):
        """
        Korean language support

        :param img_cv: image to draw
        :param text: string
        :param org: origin coordinates
        :param color: rgb tuple eg.(255,0,255)
        :param fontsize: fontsize
        :param ttf_path: default=gulim.ttf in package
        :return:
        """

        # get boundary of this text
        textsize = cv2.getTextSize(text, fontFace, 1, 2)[0]

        # get coords based on boundary
        textX = list(org)[0] - (textsize[0] // 2)
        textY = list(org)[1] + (textsize[1] // 2)

        # add text centered on image
        cv2.putText(
            img=img,
            text=text,
            org=(textX, textY),
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness
        )

    @staticmethod
    def read_fileinfo(path: str):
        """
        exit 정보를 읽는다

        :param path: image file full path
        :return: tuple(dict(exif), dict(file decription)
        """

        file_info = {}
        ctime = os.path.getctime(path)
        ftime = datetime.datetime.fromtimestamp(ctime)
        fsize = os.path.getsize(path)
        file_info['File date'] = ftime
        file_info['File size'] = fsize

        return file_info

    @staticmethod
    def plot(img,figsize=(5,5),color_mode='bgr'):
        """
        단일 이미지를 출력합니다.

        :param img: img path or bgr image or rgb image
        :param figsize: figure size
        :param color_mode: one of 'bgr', 'rgb', 'gray'(when img is path and want to plot gray img )
        :return: None
        """

        plt.figure(figsize=figsize)
        plt.xticks([])
        plt.yticks([])

        if type(img)==str and os.path.exists(img):
            img = Image.open(img)
            if color_mode=='gray':
                plt.imshow(img.convert("L"),cmap='gray')
            else:
                plt.imshow(img)
        else:
            if type(img) in [PngImagePlugin.PngImageFile,JpegImagePlugin.JpegImageFile,BmpImagePlugin.BmpImageFile]:
                plt.imshow(img)
            elif type(img)==Image.Image:
                plt.imshow(img,cmap='gray')
            else:
                if color_mode=='rgb':
                    plt.imshow(img)
                elif color_mode=='bgr' and len(img.shape)!= 2 and img.shape[2]!=1:
                    plt.imshow(img[...,::-1])
                else:
                    plt.imshow(img,cmap='gray')
        plt.show()

    @staticmethod
    def plot_imglist(path_list=None, img_list=None, url_list=None,title_list=None, cols=8, figsize=(10, 10), img_resize=(600, 600),color_mode='bgr',fontsize=12):
        """
        리스트의 이미지 경로 혹은 이미지(array) 를 받아
        출력합니다

        :param path_list: 이미지경로 리스트
        :param img_list: 이미지 리스트
        :param title_list: 위의 리스트와 길이가 같은 리스트, 사진 위의 제목으로 출력됩니다 맨앞의 글자가 @인 경우 강조됩니다.
        :param cols: 열의 개수
        :param figsize: 출력 전체 크기 (25,25) 추천
        :param img_resize: 각 이미지의 사이즈를 조정합니다.
        :param color_mode: one of 'bgr', 'rgb', 'gray'(when img is path and want to plot gray img )
        :return: None
        """
        plt.rcParams.update({'font.size': fontsize})

        if path_list is not None:
            temp_list = []
            for path in path_list:
                tmp_img = cv2.imread(path)
                if tmp_img is None:
                    continue
                resize_img = cv2.resize(tmp_img, (img_resize[0], img_resize[0]))
                temp_list.append(resize_img[..., ::-1])
            img_list = temp_list
        elif url_list is not None:
            temp_list=[]
            for url in url_list:
                if url.startswith('/Upload'):
                    url = 'https://www.py.com' + url
                response = urllib.request.urlopen(url)
                if response.getcode() == 200:
                    bytes_data = response.read()
                    tmp_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    if tmp_img is None:
                        continue
                    resize_img = cv2.resize(tmp_img, (img_resize[0], img_resize[0]))
                    temp_list.append(resize_img[..., ::-1])
                img_list = temp_list

        elif img_list is not None and color_mode == 'bgr' and len(img_list[0].shape)==3 and img_list[0].shape[2]==3:
            img_list = [v[..., ::-1] for v in img_list]

        if len(img_list) % cols == 0:
            rows = len(img_list) // cols
        else:
            rows = len(img_list) // cols + 1

        for row in range(rows):

            fig = plt.figure(figsize=figsize)

            for i in range(cols):
                idx = row * cols + i
                if idx > len(img_list) - 1:
                    break
                frame = fig.add_subplot(1, cols, i + 1)
                if title_list is not None:
                    if title_list[idx][0]=='@':
                        frame.set_title('{}'.format(title_list[idx][1:].replace(' ', '\n')), color='#FF00FF')
                    else:
                        frame.set_title('{}'.format(title_list[idx].replace(' ', '\n')), color='#808080')
                frame.set_xticks([])
                frame.set_yticks([])
                if len(img_list[idx].shape)==2 or img_list[idx].shape[2]==1:
                    frame.imshow(img_list[idx],cmap='gray')
                else:
                    frame.imshow(img_list[idx])

            plt.show()

    @staticmethod
    def get_pathlist(path, recursive=False, format=['jpg', 'png', 'jpeg'], include_secretfile=False):
        """
        디렉토리 안의 이미지들에 대한 리스트를 얻는다

        :param path: path of directory include images
        :return: images full path list, image filename list
        """
        path = os.path.abspath(path)

        path_list = []

        if recursive:
            file_list = glob.glob(path + '/**/*', recursive=recursive)
        else:
            file_list = glob.glob(path + '/*')
        extensions = format

        if os.path.isdir(path) is not True:
            return path_list

        if include_secretfile is True:
            for filename in file_list:
                if (filename.split('.')[-1]).lower() in extensions or '*' in format:
                    path_list.append(filename)
        else:
            for filename in file_list:
                if '*' in format or ((filename.split('.')[-1]).lower() in extensions and filename[0] != '.'):
                    path_list.append(filename)
        path_list.sort()

        return path_list

    @staticmethod
    def resize_image(img_cv, width=0, height=0):
        """
        비율에 맞게 리사이징 합니다. width 혹은 height 값을 인수로 받습니다. 픽셀 혹은 비율값을 받습니다.

        :param width: pixel size or ratio
        :param hegith: pixel size or ratio
        :param img_cv: cv 이미지
        :return: cv 이미지
        """

        h, w, _ = img_cv.shape

        if width == 0 and height == 0:
            return cv2.resize(img_cv, None, fx=0.5, fy=0.5)

        if (width < 1 and width != 0) or (height < 1 and height != 0):

            if height == 0:
                re_h = width
                re_w = width
                return cv2.resize(img_cv, None, fx=re_w, fy=re_h)

            elif width == 0:
                re_w = height
                re_h = height
                return cv2.resize(img_cv, None, fx=re_w, fy=re_h)

        else:

            if height == 0:
                re_h = int(width / w * h)
                re_w = width
                return cv2.resize(img_cv, (re_w, re_h))
            elif width == 0:
                re_w = int(height / h * w)
                re_h = height
                return cv2.resize(img_cv, (re_w, re_h))

        # region Modules not used frequently..

    @staticmethod
    def read_exif(data,types,image=None,bytes_data=None):
        """
        (data,types) , image , bytes_data 중 하나가 제공되어야합니다
        read exif

        :param data: data
        :param types: filepath, url, bytes 중 하나
        :param image: pillow load image data
        :param bytes_data: bytes image data
        :return:
        """
        exif={}
        if bytes_data:
            image = Image.open(BytesIO(bytes_data))
        elif data and types:
            bytes_data = PyImage.read_bytedata(data,types)
            image = Image.open(BytesIO(bytes_data))

        if image is None:
            return exif
        if image.format.upper() == 'PNG':
            return exif

        info = image._getexif()

        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_data = {}
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_data[sub_decoded] = value[t]

                    exif[decoded] = gps_data
                else:
                    exif[decoded] = value

        return exif

    @staticmethod
    def read_orientation(data=None,types=None,image=None,bytes_data=None):
        """
        (data,types) , image , bytes_data 중 하나가 제공되어야합니다
        이미지 exif 정보에 저장된 회전값을 읽어 반환합니다

        :param data: data
        :param types: filepath, url, bytes 중 하나
        :param image: pillow load image data
        :param bytes_data: bytes image data
        :return:
        """
        exif = PyImageUtil.read_exif(data,types,image,bytes_data)

        if exif is not None:
            return int(exif.get('Orientation',0))
        else:
            return 0

    @staticmethod
    def make_gridImage(path_list=None, img_list=None, shape=None, img_size=(160, 160),color_mode='bgr'):
        """
        이미지경로리스트 혹은 이미지리스트를 받아 액자식으로 구성된 하나의 이미지를 반환합니다.
        주어진 shape보다 이미지가 많을 경우 이미지리스트로 반환되며 shape가 None경우 자동으로 액자가 구성됩니다.

        :param path_list: 경로리스트
        :param img_list: 이미지리스트
        :param shape: row,col (각 이미지 개수)
        :param img_size: 각각의 이미지 크기
        :return: array or list(array)
        """



        if path_list:
            image_list = [cv2.imread(path) for path in path_list]
        elif img_list:
            if color_mode == 'bgr':
                img_list = [v[..., ::-1] for v in img_list]
            image_list = img_list

        if shape:
            col, row = shape
        else:
            col = round(len(image_list) ** (1 / 2) - 0.1)
            row = ceil(len(image_list) / col)

        mat = PyImageUtil._build_montages(image_list, img_size, (col, row))
        if len(mat) == 1:
            return mat[0]
        else:
            return mat

    @staticmethod
    def get_latlng(data=None,types=None,image=None,bytes_data=None):
        """
        (data,types) , image , bytes_data 중 하나가 제공되어야합니다
        이미지로부터 latitude, longitude 정보를 추출합니다.

        :param data: data
        :param types: filepath, url, bytes 중 하나
        :param image: pillow load image data
        :param bytes_data: bytes image data
        :return: latitude,longitude otherwise None,None
        """

        def convert_to_degress(value):
            """
            Helper function to convert the GPS coordinates
            stored in the EXIF to degress in float format
            """
            d0 = value[0][0]
            d1 = value[0][1]
            d = float(d0) / float(d1)

            m0 = value[1][0]
            m1 = value[1][1]
            m = float(m0) / float(m1)

            s0 = value[2][0]
            s1 = value[2][1]
            s = float(s0) / float(s1)

            return d + (m / 60.0) + (s / 3600.0)


        lat = None
        lng = None
        try:
            exif_data = PyImageUtil.read_exif(data,types,image,bytes_data)

            if "GPSInfo" in exif_data:
                gps_info = exif_data["GPSInfo"]
                gps_latitude = gps_info.get("GPSLatitude", None)
                gps_latitude_ref = gps_info.get('GPSLatitudeRef', None)
                gps_longitude = gps_info.get('GPSLongitude', None)
                gps_longitude_ref = gps_info.get('GPSLongitudeRef', None)
                if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
                    lat = convert_to_degress(gps_latitude)
                    if gps_latitude_ref != "N":
                        lat = 0 - lat
                    lng = convert_to_degress(gps_longitude)
                    if gps_longitude_ref != "E":
                        lng = 0 - lng
            return lat, lng
        except Exception:
            return None, None
        finally:
            return lat, lng

    @staticmethod
    def _build_montages(image_list, image_shape, montage_shape):
        """
        ---------------------------------------------------------------------------------------------
        author: Kyle Hounslow
        ---------------------------------------------------------------------------------------------
        Converts a list of single images into a list of 'montage' images of specified rows and columns.
        A new montage image is started once rows and columns of montage image is filled.
        Empty space of incomplete montage images are filled with black pixels
        ---------------------------------------------------------------------------------------------
        :param image_list: python list of input images
        :param image_shape: tuple, size each image will be resized to for display (width, height)
        :param montage_shape: tuple, shape of image montage (width, height)
        :return: list of montage images in numpy array format
        ---------------------------------------------------------------------------------------------

        example usage:

        # load single image
        img = cv2.imread('lena.jpg')
        # duplicate image 25 times
        num_imgs = 25
        img_list = []
        for i in xrange(num_imgs):
            img_list.append(img)
        # convert image list into a montage of 256x256 images tiled in a 5x5 montage
        montages = make_montages_of_images(img_list, (256, 256), (5, 5))
        # iterate through montages and display
        for montage in montages:
            cv2.imshow('montage image', montage)
            cv2.waitKey(0)

        ----------------------------------------------------------------------------------------------
        """
        if len(image_shape) != 2:
            raise Exception('image shape must be list or tuple of length 2 (rows, cols)')
        if len(montage_shape) != 2:
            raise Exception('montage shape must be list or tuple of length 2 (rows, cols)')
        image_montages = []
        # start with black canvas to draw images onto
        montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                              dtype=np.uint8)
        cursor_pos = [0, 0]
        start_new_img = False
        for img in image_list:
            if type(img).__module__ != np.__name__:
                raise Exception('input of type {} is not a valid numpy array'.format(type(img)))
            start_new_img = False
            img = cv2.resize(img, image_shape)
            # draw image to black canvas
            montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
            cursor_pos[0] += image_shape[0]  # increment cursor x position
            if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
                cursor_pos[1] += image_shape[1]  # increment cursor y position
                cursor_pos[0] = 0
                if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                    cursor_pos = [0, 0]
                    image_montages.append(montage_image)
                    # reset black canvas
                    montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                                          dtype=np.uint8)
                    start_new_img = True
        if start_new_img is False:
            image_montages.append(montage_image)  # add unfinished montage
        return image_montages


if __name__ == '__main__':

    # opencv-python의 cv2 객체와 같습니다.
    PyImageUtil.cv2

    # matplotlib의 matplotlib.pyplot 객체와 같습니다.
    PyImageUtil.plt

    # 이미지 리스트를 pdf로 변환합니다. 자세한 사용법은 docstring을 참조하세요
    PyImageUtil.images2pdf()

    # 좌표를 텍스트의 중심으로 하는 putText 입니다. 자세한 사용법은 docstring을 참조하세요
    PyImageUtil.putTextCenter()

    # 이미지 파일정보를 읽습니다. 자세한 사용법은 docstring을 참조하세요
    PyImageUtil.read_fileinfo()

    # 이미지를 출력합니다. 자세한 사용법은 docstring을 참조하세요
    PyImageUtil.plot()

    # 이미지 리스트를 타일형태로 출력합니다. 자세한 사용법은 docstring을 참조하세요
    PyImageUtil.plot_imglist()

    # 디렉토리에서 이미지 경로 리스트를 얻습니다. 자세한 사용법은 docstring을 참조하세요
    PyImageUtil.get_pathlist()

    # 이미지를 비율을 유지하며 리사이징 합니다. 자세한 사용법은 docstring을 참조하세요
    PyImageUtil.resize_image()

    # 이미지의 exif를 읽습니다. 자세한 사용법은 docstring을 참조하세요
    PyImageUtil.read_exif()

    # 이미지의 회전값을 읽습니다. 자세한 사용법은 docstring을 참조하세요
    PyImageUtil.read_orientation()

    # 이미지를 타일형태의 grid로 만듭니다. 자세한 사용법은 docstring을 참조하세요
    PyImageUtil.make_gridImage()

    # 이미지의 gps 경도,위도를 얻습니다. 자세한 사용법은 docstring을 참조하세요
    PyImageUtil.get_latlng()