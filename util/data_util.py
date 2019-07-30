# -*- coding: utf-8 -*-
"""
===============================================
data module
===============================================

========== ====================================
========== ====================================
 Module     data module
 Date       2019-03-26
 Author     hian
 Comment    `관련문서링크 <>`_
========== ====================================

*Abstract*
    * 데이터 처리관련 유틸모음

===============================================
"""


from .image_util import HianImageUtil
import os
import sys
import time
import pickle
import json
import itertools
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et
from tqdm import tqdm_notebook,tqdm
from sklearn.metrics import confusion_matrix
import pymysql
import subprocess


class HianDataUtil(object):

    def __init__(self):
        pass

    @staticmethod
    def in_notebook():
        """
        Returns ``True`` if the module is running in IPython kernel,
        ``False`` if in IPython shell or other Python shell.
        """
        return 'ipykernel' in sys.modules

    @staticmethod
    def install(name):
        """
        pip install ${name}

        :param name: pakage name
        :return: None
        """

        subprocess.call(['pip', 'install', name])


    @staticmethod
    def get_pathlist(path, recursive=False, format=['*'], include_secretfile=False):
        """
        디렉토리 안의 이미지들에 대한 리스트를 얻는다

        :param path: path of directory include images
        :return: images full path list, image filename list
        """

        return HianImageUtil.get_pathlist(path,recursive,format=format,include_secretfile=include_secretfile)

    @staticmethod
    def get_dirlist(path, include_secretfile=False):
        """
        디렉토리 리스트를 얻는다

        """
        path = os.path.abspath(path)
        dir_list = []
        file_list = os.listdir(path)

        if include_secretfile:
            for f in file_list:
                if os.path.isdir(os.path.join(path, f)):
                    dir_list.append(path + '/' + f)
        else:
            for f in file_list:
                if os.path.isdir(os.path.join(path, f)) is True and f[0] != '.':
                    dir_list.append(path + '/' + f)
        dir_list.sort()
        return dir_list

    @staticmethod
    def save_pickle(path,data):
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

    @staticmethod
    def make_histplot(value_list=None, data_dict=None, x_label='', ylabel='Quantity', title='histogram',
                      label_fontsize=7, figsize=(20, 3)):
        """
        히스토그램을 그립니다

        :param value_list:
        :param data_dict:
        :param x_label:
        :param ylabel:
        :param title:
        :param label_fontsize:
        :return:
        """
        plt.figure(figsize=figsize)

        if value_list is None:
            y = list(data_dict.values())
            y = [len(lst) for lst in y]
            x = np.arange(len(y))
            xlabel = list(data_dict.keys())
            plt.title(title)
            plt.bar(x, y)
            plt.xticks(x, xlabel, fontsize=label_fontsize)
            plt.yticks(sorted(y), fontsize=label_fontsize)
            plt.ylabel(ylabel)
            plt.show()
        else:
            y = value_list
            x = np.arange(len(y))
            plt.title(title)
            plt.bar(x, y)
            plt.xticks(x, x_label, fontsize=label_fontsize)
            plt.yticks(sorted(y), fontsize=label_fontsize)
            plt.ylabel(ylabel)
            plt.show()

    @staticmethod
    def confusion_mat(pred_list, label_list, classes, normalize=False, title='Confusion matrix', figsize=(5, 5)):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.

        """

        cm = confusion_matrix(label_list, pred_list)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
