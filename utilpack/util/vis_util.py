# -*- coding: utf-8 -*-
"""
===============================================
vis_util module
===============================================

========== ====================================
========== ====================================
 Module     vis_util module
 Date       2019-03-26
 Author     heewinkim
 Comment    `관련문서링크 <>`_
========== ====================================

*Abstract*
    * 데이터의 시각화 관련 유틸 모음

===============================================
"""


import matplotlib.pyplot as plt
import numpy as np
import cv2


class PyVisUtil(object):

    @staticmethod
    def color_hist(cv_img):
        """
        show color histogram on 2D plot

        :param cv_img:
        :return:None
        """

        chans = cv2.split(cv_img)
        colors = ("b", "g", "r")
        plt.figure()
        plt.title("Color Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixcels")
        features = []

        for (chan, color) in zip(chans, colors):
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            features.extend(hist)

            plt.plot(hist, color=color)
            plt.xlim([0, 256])

        print("Flattened feature vector size: %d " % (np.array(features).flatten().shape))
        plt.show()

    @staticmethod
    def gray_hist(gray_img):
        """
        show gray histogram

        :param gray_img: cv gray image
        :return: None
        """

        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixcels")
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()
