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
 Comment    `관련문서링크 <call to heewinkim >`_
========== ====================================

*Abstract*
    * keras 관련 유틸 모음

===============================================
"""



import os
import sys
current_dir = os.path.dirname(__file__) # common.util
parent_dir = os.path.dirname(current_dir) # common
sys.path.insert(0,parent_dir)
import cv2
import keras
import numpy as np
import pandas as pd
from time import time
import keras.backend as K
import matplotlib.pyplot as plt
import requests
import json


class HianKerasUtil(object):


    @staticmethod
    def gen2list(generator,step):

        x_list = []
        y_list = []

        for i,(x,y) in enumerate(generator):
            if i>step:
                break
            x_list.extend(list(x))
            y_list.extend(list(y))

        return x_list,y_list

    @staticmethod
    def getGenerator_fromDataframe(anno_path, batch_size, input_shape, class_mode, validation_split=0.2,
                                   rescale=1. / 255,color_mode='rgb'):
        """
        annotation 파일을 읽어 학습 제너레이터 혹은 학습,검증 제너레이터를 반환합니다
        리턴값은 validation_split 값에 따라 달라집니다.


        :param anno_path: annotation 파일 경로
        :param batch_size: batch size
        :param input_shape: input shape (ex. [64,64,3])
        :param class_mode: 'categorical' or ','detection'
        :param validation_split: if 0, return train generator otherwire 0~1.0 return train,validation generator
        :param rescale: rescaling (ex. 1./255)
        :param color_mode: 'rgb','grayscale' 두가지만 제공됩니다.
        :return: train_generator or train_generator,validation_generator depend on validation_split value

        """

        if class_mode == 'categorical':
            input_headers = 'label'
            train_df = pd.read_csv(anno_path, dtype=str)
        elif class_mode == 'detection':
            class_mode = 'other'
            train_df = pd.read_csv(anno_path)
            input_headers = list(train_df)[1:]
        else:
            class_mode = 'other'
            train_df = pd.read_csv(anno_path)
            input_headers = list(train_df)[1:]

        if validation_split!=0:
            datagen = keras.preprocessing.image.ImageDataGenerator(validation_split=validation_split, rescale=rescale)
            train_generator = datagen.flow_from_dataframe(train_df,
                                                          target_size=input_shape[:2],
                                                          x_col='imgPath',
                                                          y_col=input_headers,
                                                          batch_size=batch_size,
                                                          class_mode=class_mode,
                                                          subset='training',
                                                          color_mode=color_mode
                                                          )
            validation_generator = datagen.flow_from_dataframe(train_df,
                                                               target_size=input_shape[:2],
                                                               x_col='imgPath',
                                                               y_col=input_headers,
                                                               batch_size=batch_size,
                                                               class_mode=class_mode,
                                                               subset='validation',
                                                               color_mode=color_mode
                                                               )

            return train_generator, validation_generator
        else:
            datagen = keras.preprocessing.image.ImageDataGenerator(rescale=rescale)
            generator = datagen.flow_from_dataframe(train_df,
                                                          target_size=input_shape[:2],
                                                          x_col='imgPath',
                                                          y_col=input_headers,
                                                          batch_size=batch_size,
                                                          class_mode=class_mode,
                                                          color_mode=color_mode
                                                          )
            return generator

    def getGenerators(self,root_path, input_resolution=(224, 224), batch_size=64,rescale=1./255,class_mode='categorical'):
        """
        root_path 아래에 각 train,validation,test 데이터에 대한 제너레이터를 제공합니다.
        폴더가 존재하지 않는경우 해당 제너레이터는 None값으로 반환됩니다.

        :param path: directory include images
        :param input_resolution: ex. (224,224)
        :param batch_size: ex.64
        :param rescale: ex.1./255
        :param class_mode: 'categorical','binary'
        :raise RUNTIME_ERROR:
        :return: trian_gen,validation_gen,test_gen
        """

        train_dir = root_path + '/train'
        validation_dir = root_path + '/validation'
        test_dir = root_path + '/test'

        generators=[]
        for path in [train_dir,validation_dir,test_dir]:
            if os.path.exists(path):
                generators.append(HianKerasUtil.getGenerator_fromDirectory(path,input_resolution,batch_size,rescale,class_mode))
            else:
                generators.append(None)

        return generators

    @staticmethod
    def getGenerator_fromDirectory(path, input_resolution=(224, 224), batch_size=64,rescale=1./255,class_mode='categorical'):
        """
        directory 에 저장된 이미지들에 대한 제너레이터를 반환합니다.
        only support color_model rgb

        :param path: directory include images
        :param input_resolution: ex. (224,224)
        :param batch_size: ex.64
        :param rescale: ex.1./255
        :raise RUNTIME_ERROR:
        :return: generator
        """
        # 데이터 준비
        train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=rescale)

        if os.path.exists(path):
            generator = train_datagen.flow_from_directory(
                path,
                target_size=input_resolution,
                batch_size=batch_size,
                class_mode=class_mode)
            return generator
        else:
            raise HianError(ERROR_TYPES.RUNTIME_ERROR,'{} is not exist'.format(path))

    @staticmethod
    def get_generatorStep(generator):
        """
        return data step N

        :return: int,step
        """
        step = generator.n // generator.batch_size
        return step

    @staticmethod
    def test_model(model, img_list=None, label_list=None, test_generator=None, cols=4, figsize=(10, 10), steps=1,
                   rescale=None, color_mode='rgb'):
        """
        test_generator.batch_size >= rows*cols

        """
        x_list = []
        y_list = []
        input_shape = tuple(np.array(model.input.shape[1:]).astype(int))

        if img_list is None:

            for _ in range(steps):
                try:
                    x, y = test_generator.next()
                except:
                    x = test_generator.next()

                x_list.extend(x)
                y_list.extend(y)

        elif test_generator is None:

            if color_mode == 'rgb':
                img_list = [v[..., ::-1] for v in img_list]
            if rescale:
                img_list = [v * rescale for v in img_list]

            img_list = [cv2.resize(v, (input_shape[0], input_shape[1])) for v in img_list]

            x_list.extend(img_list)

            if label_list is None:
                y_list.extend([-1 for _ in range(len(img_list))])
            else:
                y_list.extend(label_list)

        rows = len(x_list) // cols + 1

        for row in range(rows):

            fig = plt.figure(figsize=figsize)

            for i in range(cols):

                idx = row * cols + i
                if idx >= len(x_list):
                    break

                tic = time()
                x_ = x_list[idx].reshape((1, input_shape[0], input_shape[1], input_shape[2]))
                output = model.predict(x_)
                toc = time()

                frame = fig.add_subplot(1, cols, i + 1)
                predict_idx = np.argmax(output[0])
                if y_list[idx] != -1:
                    title = 'T:{} P:{} - {}ms\n{:.2f}%'.format(np.argmax(y_list[idx]), predict_idx,
                                                               int(((toc - tic) * 1000)), output[0][predict_idx] * 100)

                    if np.argmax(y_list[idx]) == predict_idx:
                        frame.set_title(title, fontsize=figsize[0], color='#808080')
                    else:
                        frame.set_title(title, fontsize=figsize[0], color='#FF00FF')

                else:
                    title = 'Pred:{} - {}ms\n{:.2f}%'.format(predict_idx, int(((toc - tic) * 1000)),
                                                             output[0][predict_idx] * 100)

                    frame.set_title(title, fontsize=figsize[0], color='#808080')

                frame.set_xticks([])
                frame.set_yticks([])
                frame.imshow(x_list[idx])

            plt.show()

    @staticmethod
    def show_generator(generator, rows=1, cols=4, figsize=(10, 10),rescale=None,color_mode='bgr'):


        x, y = generator.next()

        if color_mode=='bgr':
            x = [v[...,::-1] for v in x]

        if rescale:
            x = [ v * rescale for v in x ]

        for row in range(rows):

            fig = plt.figure(figsize=figsize)

            for i in range(cols):
                idx = row * cols + i

                frame = fig.add_subplot(1, cols, i + 1)
                frame.set_title('label : {}'.format(np.argmax(y[idx])), color='#808080')
                frame.imshow(x[idx])

            plt.show()

    @staticmethod
    def plot_history(history):
        # 그래프 확인
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        loss_ax.plot(history.history['loss'], 'y', label='train loss')
        loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        loss_ax.legend(loc='upper left')

        acc_ax.plot(history.history['acc'], 'b', label='train acc')
        acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
        acc_ax.set_ylabel('accuracy')
        acc_ax.legend(loc='lower left')

        plt.show()

    @staticmethod
    def evaluate_model(test_generator, model,print_=True,use_remote=False,**kwargs):
        """
        generator 를 입력받아 모델 테스트를 진행합니다.

        :param test_generator: 제너레이터
        :param model: 테스트할 모델객체
        :param print_: bool 결과를 print 할것인지
        :param use_remote: remote post 를 사용할지
        :param ip: use_remote가 참인 경우, post 할 ip
        :param port: use_remote가 참인 경우, post 할 port
        :param url: use_remote가 참인 경우, post 할 url 경로

        :return: score( tuple(loss,acc) )
        """
        try:
            test_steps = test_generator.n // test_generator.batch_size
            score = model.evaluate_generator(test_generator, steps=test_steps)
            if print_:
                print()
                print("loss: {}%".format(round(score[0], 3)), end='\t')
                print("acc: {}%".format(round(score[1], 3) * 100), end='\n\n')

            if use_remote:

                port = kwargs.get('port',9000)
                ip = kwargs.get('ip','0.0.0.0')
                url = kwargs.get('url','/')

                send = {'loss':score[0],'acc':score[1]}

                requests.post('{}:{}{}'.format(ip,port,url),{'data':json.dumps(send)})

            return score
        except Exception:
            raise HianERROR(ERROR_TYPES.RUNTIME_ERROR,'Need to compile model')

    @staticmethod
    def show_cam(generator, model, show_layer_idx=-3,figsize=(10,10),show=False,max_idx=16):
        ## read image and preprocess it
        x, y = generator.next()
        img_arr = []
        cams = []
        predicts = []

        for idx, img in enumerate(x):

            if idx>=max_idx:
                break

            class_idx = np.argmax(y[idx])

            img_array = np.expand_dims(img, axis=0)

            ## get prediction result and conv_output values
            get_output = K.function([model.layers[0].input], [model.layers[show_layer_idx].output, model.layers[-1].output])
            [conv_outputs, predictions] = get_output([img_array])
            conv_outputs = conv_outputs[0, :, :, :]
            class_weights = model.layers[-1].get_weights()[0]

            ## calculate cam
            cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
            for i, w in enumerate(class_weights[:, class_idx]):
                cam += w * conv_outputs[:, :, i]

            cam /= np.max(cam)
            cam = cv2.resize(cam, (x[0].shape[0], x[0].shape[1]))

            img_arr.append(img)
            cams.append(cam)
            predicts.append(predictions[0])
            labels = y

        if show:
            for img,lbl,cam,pred in zip(img_arr,labels,cams,predicts):

                for row in range(1):

                    fig = plt.figure(figsize=figsize)

                    for i in range(8):

                        frame = fig.add_subplot(1, 8, i + 1)
                        frame.set_title('label : {}'.format(np.argmax(lbl)), color='#808080')
                        frame.imshow(img)

                    plt.show()

                for row in range(1):

                    fig = plt.figure(figsize=figsize)

                    for i in range(8):

                        frame = fig.add_subplot(1, 8, i + 1)
                        frame.set_title('predict : {}'.format(np.argmax(pred)), color='#808080')
                        frame.imshow(cam)

                    plt.show()
        else:
            return img_arr, labels, cams, predicts

    @staticmethod
    def save_model(save_model_name, save_model, mode=1):
        """
        모델을 저장한다

        :param save_model_name:
        :param save_model:
        :param mode:
                 0 : save in onefile
                1 : save weight,achitecture sperately
        :return:
        """
        if mode == 0:
            # 모델 구조, 가중치, 학습파라미터 모두 저장
            save_model.save('{}.h5'.format(save_model_name))

        elif mode == 1:
            # 모델 구조 저장
            model_json = save_model.to_json()

            with open("{}.json".format(save_model_name), "w") as json_file:
                json_file.write(model_json)
            # 모델 weight 저장
            save_model.save_weights("{}.h5".format(save_model_name))

    @staticmethod
    def get_callback(use_ckpt=True, ckpt_path='./models',ckpt_filename='weights.{epoch:02d}-{val_loss:.3f}.hdf5',
                     use_early_stop=True, early_stop_patience=3,
                     use_learning_rate_scheduler=False, lr_rate=0.001, lr_decay=0.9,
                     use_csv_logger=False, csv_save_dir='./csv_log',
                     use_tensorboard=False, tb_save_dir='./tb_log',
                     use_reduceLROnPlateau=False, reduce_lr_patience=3,
                     use_remote_monitor=False, remote_port=9000):
        """
        콜백 함수를 정의, 콜백함수 리스트를 반환


        :param use_ckpt:
        :param ckpt_path:
        :param use_early_stop:
        :param early_stop_patience:
        :param use_learning_rate_scheduler:
        :param lr_rate:
        :param lr_decay:
        :param use_csv_logger:
        :param csv_save_dir:
        :param use_tensorboard:
        :param tb_save_dir:
        :param use_reduceLROnPlateau:
        :param reduce_lr_patience:
        :return:
        """

        callbacks = []

        if use_ckpt:
            ckpt = keras.callbacks.ModelCheckpoint(ckpt_path + '/'+ckpt_filename, monitor='val_loss',
                                   save_best_only=True)
            callbacks.append(ckpt)
        if use_early_stop:
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=early_stop_patience)
            callbacks.append(early_stopping)
        if use_learning_rate_scheduler:
            learning_rate_scheduler = keras.callbacks.LearningRateScheduler(schedule=lambda epoch: lr_rate * (lr_decay ** epoch))
            callbacks.append(learning_rate_scheduler)
        if use_csv_logger:
            csv_logger = keras.callbacks.CSVLogger(csv_save_dir + '/log.csv')
            callbacks.append(csv_logger)
        if use_tensorboard:
            tensorboard = keras.callbacks.TensorBoard(log_dir=tb_save_dir)
            callbacks.append(tensorboard)
        if use_reduceLROnPlateau:
            reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=reduce_lr_patience)
            callbacks.append(reduce_lr_on_plateau)
        if use_remote_monitor:
            remote_monitor = keras.callbacks.RemoteMonitor(root='http://localhost:{}'.format(remote_port))
            callbacks.append(remote_monitor)

        return callbacks

def example():

    # 파라미터 정의
    batch_size = 64
    input_shape = (224, 224, 3)
    n_classes = 3
    epoch = 100
    data_dir = './data'

    # 학습 데이터 파싱
    train_gen, validation_gen, test_gen = HianKerasUtil.getGenerators(data_dir, input_shape[:2], batch_size)
    train_step = HianKerasUtil.get_generatorStep(train_gen)
    validation_step = HianKerasUtil.get_generatorStep(validation_gen)
    test_step = HianKerasUtil.get_generatorStep(test_gen)


    # 데이터 이미지 출력
    HianKerasUtil.show_generator(train_gen)


    # 모델 구현
    from keras.applications.vgg16 import VGG16
    from keras.layers import GlobalAveragePooling2D,Dense
    from keras.models import Model
    from keras.optimizers import Adam
    backbone = VGG16(include_top=False,weights='imagenet',input_shape=input_shape)
    x = backbone.output
    x = GlobalAveragePooling2D(x)
    output = Dense(n_classes,activation='softmax')(x)
    model = Model(inputs=backbone.input,outputs=output)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # 가중치 로드
    model.load_weights('./checkpoint/checkpoint-22-0.0097.h5')

    # 콜백함수 정의
    callbacks = HianKerasUtil.get_callback()

    # 학습
    history = model.fit_generator(
            generator= train_gen,
            steps_per_epoch=train_step,
            epochs=epoch,
            validation_data=validation_gen,
            validation_steps=validation_step,
            callbacks=callbacks,
            verbose=1
            )

    # 최종 학습모델 저장
    HianKerasUtil.save_model(save_model_name='good_model',save_model=model,mode=0)


    # 학습 그래프 확인
    HianKerasUtil.plot_history(history)


    # 정확도 확인
    HianKerasUtil.evaluate_model(test_gen,model)


    # 각 이미지별 예측 결과 상세 확인
    HianKerasUtil.test_model(test_gen, model, cols=8, steps=1, figsize=(20, 20))


    # CAM 확인
    HianKerasUtil.show_cam(test_gen,model,show_layer_idx=-5,figsize=(20,20),max_idx=16)
