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
from .vis_util import PyVisUtil
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
from pdfrw import PdfReader, PdfWriter, PdfDict, PdfName


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


class PyDataUtil(object):

    pdf = Pdf

    @staticmethod
    def comprehension(list, attr):
        return [v[attr] for v in list]

    @staticmethod
    def plot3D(cluster_list, label_list=None, figsize=(15, 15), colors=["#ff0000", "#0000ff", "#00ff00"],
               seperate_plot=False):
        """
        클러스터링 리스트를 3d 공간에 뿌립니다. 입력의 cluster_list 는 아래와 같은 조건이어야 합니다.
        (N,M,3) vector / N = 클러스터 개수, M = 클러스터의 표본개수 , 3 = 벡터값(x,y,z)
        다차원 클러스터 요소인 경우
        아래와 같이 PCA를 통해 3d공간에 맵핑할 수 있습니다.

            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            pca.fit(vector_list)
            vectors = pca.transform(vector_list)

        :param cluster_list: (N,M,3) vector / N = 클러스터 개수, M = 클러스터의 표본개수 , 3 = 벡터값(x,y,z)
        :param label_list: label list
        :param figsize: plot size
        :param colors: 색상 지정 (eg. #ff00ff
        :param seperate_plot: 클러스터마다 따로 출력 , default = False
        :return:
        """
        PyVisUtil.plot3D(cluster_list,label_list,figsize,colors,seperate_plot)

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
    def query2mysql_without_timeout(query, host, port, user, password, db, charset='utf8', to_dataframe=False, timeout=60):
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
                curs.execute(query)
                rows = curs.fetchall()
        finally:
            conn.close()

        if not len(rows):
            return None
        elif to_dataframe:
            import pandas as pd
            return pd.DataFrame(rows)
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
            while len(data) > 0:
                dst.write(data)
                data = src.read(1024)

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

    @staticmethod
    def pie(data, labels=None, title='', save_path=None, legend=True, radius=1, explode=0.05, shadow=True, colors=None,
            figsize=(8, 5), fontsize=12):
        """
        데이터의 파이를 그립니다.
        기본적으로 각 elelment의 percentage 및 개수를 표현하며, legend가 달립니다.
        ex. pie(data=[1,2,3],labels=['a','b','c'])

        :param data: 어떠한 데이터의 수량을 나타내는 리스트
        :param save_path: 경로 제공시 pie 이미지를 저장합니다.(eg. path/to/pie.png)
        :param title: 제목
        :param labels: 라벨 제공시 각 엘리먼트의 이름이 표시됩니다. 라벨은 데이터개수만큼의 문자요소 리스트 이어야 합니다.
        :param radius: default 1, matplotlib.pyplot.pie의 radius와 같습니다.
        :param explode: default 0.05, matplotlib.pyplot.pie의 요소와 같지만 리스트가 아닌 단일 float값을 받습니다.
        :param shadow: boolean 그림자 표시여부
        :param colors: default None, 각 요소의 색깔을 지정하고 싶을때 요소길이 만큼의 컬러가 주어져야합니다.
        :param figsize: figure size
        :param fontsize: element의 값을 표시하는 폰트의 크기
        :return:
        """

        data = list(data)
        labels = list(labels) if labels else None

        def label_func(percent, data):
            absolute = int(percent / 100. * np.sum(data))
            return "{:.1f}%\n({:d})".format(percent, absolute)

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(aspect="equal"))

        if not labels:
            labels = [str(v) for v in data]

        wedges, texts, autotexts = ax.pie(
            data, autopct=lambda pct: label_func(pct, data), textprops=dict(color="w", fontsize=fontsize),
            radius=radius, explode=[explode] * len(data), shadow=shadow, colors=colors, labels=labels)

        if legend:
            ax.legend(wedges, labels, title="Label", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=fontsize, weight="bold")
        ax.set_title(title)
        if save_path:
            plt.savefig('{}'.format(save_path))
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

    # 기본적인 셋팅이 되어있는 matplotlib.pie 입니다. 자세한 사용법은 docstring을 참조하세요
    PyDataUtil.pie()

    # pdf 파일에 메타데이터를 추가합니다.
    PyDataUtil.pdf.add_metadata('file.pdf','save.pdf',{'a':1,'b':2,'c':3},'keyname')

    # pdf 파일에 있는 메타데이터를 읽어옵니다.
    metadata = PyDataUtil.pdf.read_metadata('file.pdf','keyname')
    print(metadata)  # {'a': 1, 'b': 2, 'c': 3}