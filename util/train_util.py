# -*- coding: utf-8 -*-
"""
===============================================
keras_tool module
===============================================

========== ====================================
========== ====================================
 Module     keras_tool module
 Date       2019-03-26
 Author     hian
========== ====================================

*Abstract*
    * 학습 관련 유틸 모음

===============================================
"""

from .image_util import PyImageUtil
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from random import randint
import matplotlib.pyplot as plt
from collections import OrderedDict


class PyTrainUtil(object):

    @staticmethod
    def plot_testlist(path_list,pred_list,conf_list,label_list=None,cols=8,figsize=(10,10),img_resize=(600,600),miss_only=False):

        if miss_only:
            path_list_ = []
            pred_list_ = []
            conf_list_ = []
            label_list_ = []
            if label_list is None:
                label_list = [ 0 for _ in range(len(pred_list))]

            for path,pred,label,conf in zip(path_list,pred_list,label_list,conf_list):
                if pred!=label:
                    path_list_.append(path)
                    pred_list_.append(pred)
                    label_list_.append(label)
                    conf_list_.append(conf)
            path_list = path_list_
            pred_list = pred_list_
            label_list = label_list_
            conf_list = conf_list_


        title_list = ['label={} pred={} conf={:.2f}'.format(label, pred, conf) for label, pred, conf in
                      zip(label_list, pred_list, conf_list)]

        temp_list=[]
        for path in path_list:
            tmp_img = cv2.imread(path)
            if tmp_img is None:
                continue
            resize_img = cv2.resize(tmp_img,(img_resize[0],img_resize[0]))
            temp_list.append(resize_img[...,::-1])
        img_list = temp_list

        if len(img_list) % cols == 0:
            rows = len(img_list) // cols
        else:
            rows = len(img_list) // cols + 1

        for row in range(rows):
            fig = plt.figure(figsize=figsize)

            for i in range(cols):
                idx = row * cols + i
                if idx>len(img_list)-1:
                    break
                frame = fig.add_subplot(1, cols, i + 1)


                if pred_list[idx]==label_list[idx]:
                    frame.set_title(title_list[idx].replace(' ','\n'), color='#808080')
                else:
                    frame.set_title(title_list[idx].replace(' ','\n'), color='#FF00FF')

                frame.set_xticks([])
                frame.set_yticks([])
                frame.imshow(img_list[idx])

            plt.show()

    @staticmethod
    def get_testset(path,shuffle=True):
        """
        각 라벨들에 대한 디렉토리가 포함된 경로를 입력받아
        path_list, label_list 형태로 반환합니다.
        각 라벨에 해당하는 디렉토리에는 이미지 데이터들이 들어있습니다.

        :param path: 각 라벨들의 디렉토리를 포함하는 경로
        :param shuffle: 데이터 셔플 여부
        :return: path_list,label_list
        """

        dir_list = PyImageUtil.get_pathlist(path)
        name_to_label=OrderedDict()

        path_list = []
        label_list = []

        for i, directory in enumerate(dir_list):

            label_name = directory.split('/')[-1]
            name_to_label[label_name]=i

            path_list_ = os.listdir(directory)
            path_list_ = [ path for path in path_list_ if path[0] != '.']
            path_list_ = [ directory+'/'+filename for filename in path_list_]
            path_list.extend(path_list_)
            label_list.extend([ i for _ in range(len(path_list_))])

        if shuffle:
            new_path_list = []
            new_label_list = []

            while len(path_list) > 0:
                random_index = randint(0, len(path_list) - 1)
                new_path_list.append(path_list.pop(random_index))
                new_label_list.append(label_list.pop(random_index))

            path_list = new_path_list
            label_list = new_label_list

        print('loaded {} images , {} classes'.format(len(path_list),len(name_to_label)))
        for k,v in name_to_label.items():
            print('{} : {}'.format(k,v),end='   ')
        print()

        return path_list,label_list

    @staticmethod
    def make_npy_dataset(path,save_path='.',file_format='jpg',input_shape=(224,224,3)):
        """
        npy 형식의 이미지 데이터셋 파일을 생성합니다.

        :param path: train,validation,test를 하위디렉토리도 둔 경로(train 디렉토리는 필수)
        :param save_path: 각 데이터셋.npy가 저장될 디렉토리경로
        :param file_format: image 포맷
        :param input_shape: e.g(224,224,3)
        :return: None
        """

        train_data=[]
        train_label=[]
        validation_data=[]
        validation_label=[]
        test_data=[]
        test_label=[]

        if not os.path.exists(path + '/train'):
            print('train data must needed!')
            return -1

        label_list = (path + '/train')
        label_list = [path_.split('/')[-1] for path_ in label_list]
        label_dict = {}
        for i, label in enumerate(label_list):
            label_dict[label] = i

        if os.path.exists(path+'/train'):
            train_list = glob.glob(path+'/train/**/*.{}'.format(file_format),recursive=True)
            print('N train list : {}'.format(len(train_list)))
            print("processing train data...")
            for img_path in tqdm(train_list):
                img = cv2.imread(img_path)
                img = cv2.resize(img,input_shape[:2])
                train_data.append(img)
                train_label.append(label_dict[img_path.split('/')[-2]])
            train_data = np.array(train_data)
            train_label = np.array(train_label)

        if os.path.exists(path+'/validation'):
            validation_list = glob.glob(path+'/validation/**/*.{}'.format(file_format),recursive=True)
            print('N validation list : {}'.format(len(validation_list)))
            print("processing validation data...")
            for img_path in tqdm(validation_list):
                img = cv2.imread(img_path)
                img = cv2.resize(img, input_shape[:2])
                validation_data.append(img)
                validation_label.append(label_dict[img_path.split('/')[-2]])
            validation_data = np.array(validation_data)
            validation_label = np.array(validation_label)

        if os.path.exists(path+'/test'):
            test_list = glob.glob(path+'/test/**/*.{}'.format(file_format),recursive=True)
            print('N test list : {}'.format(len(test_list)))
            print("processing test data...")
            for img_path in tqdm(test_list):
                img = cv2.imread(img_path)
                img = cv2.resize(img, input_shape[:2])
                test_data.append(img)
                test_label.append(label_dict[img_path.split('/')[-2]])
            test_data = np.array(test_data)
            test_label = np.array(test_label)

        np.save(save_path+'/train_data',train_data)
        np.save(save_path + '/train_label', train_label)
        np.save(save_path + '/validation_data', validation_data)
        np.save(save_path + '/validation_label', validation_label)
        np.save(save_path + '/test_data', test_data)
        np.save(save_path + '/test_label', test_label)

        print('Save data to npy is Done.')
