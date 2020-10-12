# -*- coding: utf-8 -*-
"""
===============================================
data module
===============================================

========== ====================================
========== ====================================
 Module     data module
 Date       2019-03-26
 Author     heewinkim
========== ====================================

*Abstract*
    * 데이터 처리관련 유틸모음

===============================================
"""


from .image_util import PyImageUtil
from .time_util import Timeout
import os
import sys
import json
import pickle
import timeit
import itertools
import subprocess
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import pymysql
from tqdm.auto import tqdm


class PyDataUtil(object):

    @staticmethod
    def comprehesion(list, attr):
        return [v[attr] for v in list]

    @staticmethod
    def plot3D(arr_list, label_list=None, figsize=(15, 15), colors=["#ff0000", "#0000ff", "#00ff00"],
               seperate_plot=False):
        from mpl_toolkits.mplot3d import Axes3D
        if not seperate_plot:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

            ax.set_xlabel('x', fontsize=15)
            ax.set_ylabel('y', fontsize=15)
            ax.set_zlabel('z', fontsize=15)

            for i, data in enumerate(arr_list):
                ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors[i % len(colors)], s=30)
            if label_list:
                ax.legend(label_list)
            else:
                ax.legend(['data{}'.format(i) for i in range(len(arr_list))])
            ax.grid()
        else:
            figsize = (list(figsize)[1] * len(arr_list), list(figsize)[1])
            fig = plt.figure(figsize=figsize)

            for i, data in enumerate(arr_list):
                ax = fig.add_subplot(1, len(arr_list), i + 1, projection='3d')

                ax.set_xlabel('x', fontsize=15)
                ax.set_ylabel('y', fontsize=15)
                ax.set_zlabel('z', fontsize=15)

                ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors[i % len(colors)], s=30)

                if label_list:
                    ax.legend([label_list[i]])
                else:
                    ax.legend(['data{}'.format(i)])
                ax.grid()
        plt.show()

    @staticmethod
    def query2mysql(query,host,port,user,password,db,charset='utf8',to_dataframe=False,timeout=10):
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            db=db,
            charset=charset,
            cursorclass=pymysql.cursors.DictCursor)
        try:
            with conn.cursor() as curs:
                with Timeout(seconds=timeout):
                    curs.execute(query)
                rows = curs.fetchall()
        finally:
            conn.close()

        if not len(rows):
            return None
        elif to_dataframe:
            import pandas as pd
            return pd.DataFrame().from_dict(rows)
        else:
            return rows

    @staticmethod
    def save_json(data,path):
        with open(path,'w',encoding='utf8') as f:
            f.write(json.dumps(data))

    @staticmethod
    def load_json(path):
        return json.loads(open(path,encoding='utf8').read())

    @staticmethod
    def download(url:str,save_path):

        print('download {} ..'.format(url.split('/')[-1]))
        with urllib.request.urlopen(url) as src, open(save_path, 'wb') as dst:
            data = src.read(1024)
            pbar = tqdm(total=int(np.ceil(src.length / 1024)))
            while len(data) > 0:
                pbar.update(1)
                dst.write(data)
                data = src.read(1024)
            pbar.close()

    @staticmethod
    def func_test(f, number=1000, time_unit='ms', repeat=3, verbose=True, *args, **kwargs):
        def wrapper(func, *args, **kwargs):
            def wrapped():
                return func(*args, **kwargs)

            return wrapped

        wrappered1 = wrapper(f, *args, **kwargs)
        times = timeit.repeat(wrappered1, repeat=repeat, number=number)

        times = sum(times) / len(times)

        if time_unit == 'ms':
            times = int(times * 1000)
        elif time_unit == 'us':
            times = int(times * 1000000)

        if verbose:
            print(times, time_unit)

        return times

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

        return PyImageUtil.get_pathlist(path,recursive,format=format,include_secretfile=include_secretfile)

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
        from sklearn.metrics import confusion_matrix

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


if __name__ == '__main__':

    # 3차원 데이터 리스트를 받아 출력합니다. 자세한 사용법은 docstring을 참조하세요.
    PyDataUtil.plot3D()

    # mysql에 쿼리를 날립니다. 자세한 사용법은 docstring을 참조하세요.
    PyDataUtil.query2mysql()

    # dict 데이터를 json으로 저장합니다. 자세한 사용법은 docstring을 참조하세요.
    PyDataUtil.save_json()

    # 저장된 Json을 dict로 읽습니다. 자세한 사용법은 docstring을 참조하세요.
    PyDataUtil.load_json()

    # 주어진 url의 데이터를 다운로드 합니다. 자세한 사용법은 docstring을 참조하세요.
    PyDataUtil.download()

    # 함수를 테스트 합니다. 자세한 사용법은 docstring을 참조하세요.
    PyDataUtil.func_test()

    # 주피터 노트북 환경인지를 boolean값으로 리턴합니다.
    is_in_notebook = PyDataUtil.in_notebook()
    print(is_in_notebook)  # False

    # pypi의 패키지를 설치합니다.
    PyDataUtil.install('setuptools')

    # 디렉토리의 특정 파일포맷 리스트를 얻습니다.  자세한 사용법은 docstring을 참조하세요.
    PyDataUtil.get_pathlist('path/to/dir',recursive=False,format=['txt'])

    # 경로의 디렉토리 리스트를 얻습니다. 자세한 사용법은 docstring을 참조하세요.
    PyDataUtil.get_dirlist()

    # 데이터를 pickle로 저장합니다. 자세한 사용법은 docstring을 참조하세요.
    PyDataUtil.save_pickle({'example':'data'},'data.pkl')

    # 저장된 pickle 데이터를 불러옵니다. 자세한 사용법은 docstring을 참조하세요.
    data = PyDataUtil.load_pickle('data.pkl')
    print(data)  # {'example':'data'}

    # 데이터 리스트를 받아 히스토그램으로 출력합니다. 자세한 사용법은 docstring을 참조하세요.
    PyDataUtil.make_histplot()

    # 데이터를 받아 confusion matrix를 출력합니다. 자세한 사용법은 docstring을 참조하세요.
    PyDataUtil.confusion_mat()