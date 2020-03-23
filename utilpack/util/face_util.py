# -*- coding: utf-8 -*-
"""
===============================================
face_util module
===============================================

========== ====================================
========== ====================================
 Module     face_util module
 Date       2019-07-29
 Author     heewinkim
========== ====================================

*Abstract*
    * PyFaceUtil 클래스 제공 - 메서드 (get_align_faces,vis_face_landmarks
    * shape_predictor 모델 다운로드 경로

===============================================
"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__)) # common.util
parent_dir = os.path.dirname(current_dir) # common
sys.path.insert(0,parent_dir)
import cv2


class PyFaceUtil(object):

    @staticmethod
    def crop_face(image, x1, y1, x2, y2, copy=True):
        """
        cropped 된 얼굴을 반환합니다.

        Overloaded function list.

        :param image: image
        :param x1,y1,x2,y2: face's coordinates, each points are left,top,right,bottom.
        :param copy: return copy data, default True


        :return: cropped face
        """
        if copy:
            return image[y1:y2,x1:x2,:].copy()
        else:
            return image[y1:y2, x1:x2, :]

    @staticmethod
    def draw_face(image, x1, y1, x2, y2, color=(255, 0, 255), thickness=3):
        """
        img_cv 위에 face 사각형을 그립니다.

        :param image: image
        :param x1,y1,x2,y2: face's coordinates, each points are left,top,right,bottom.
        :return: None
        """
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)