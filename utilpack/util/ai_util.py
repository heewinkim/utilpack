# -*- coding: utf-8 -*-
"""
===============================================
ai_util module
===============================================

========== ====================================
========== ====================================
 Module     ai_util module
 Date       2020-05-29
 Author     heewinkim
========== ====================================

*Abstract*
    * 시간 관련 유틸 모음

===============================================
"""

import os


class tf(object):

    class v1(object):
        pass

    class v2(object):

        @staticmethod
        def kerasmodel2frozengraph(model, save_path, printLayers=False):
            """

            :param model: keras model 객체
            :param save_path:
            :return:
            """

            import tensorflow as tf

            if not tf.__version__ >= '2.0.0':
                print('Required tensorflow>=2.0.0.')
                return

            if not len(model.inputs) == 1:
                print('Only Support One model input')
                return

            full_model = tf.function(lambda x: model(x))
            full_model = full_model.get_concrete_function(
                tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

            # Get frozen ConcreteFunction
            from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
            frozen_func = convert_variables_to_constants_v2(full_model)
            frozen_func.graph.as_graph_def()
            if printLayers:
                layers = [op.name for op in frozen_func.graph.get_operations()]
                print("-" * 50)
                print("Frozen model layers: ")
                for layer in layers:
                    print(layer)

            print("-" * 50)
            print("Frozen model inputs: ")
            print(frozen_func.inputs)
            print("Frozen model outputs: ")
            print(frozen_func.outputs)

            # Save frozen graph from frozen ConcreteFunction to hard drive
            tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                              logdir='/'.join(save_path.split('/')[:-1]),
                              name=save_path.split('/')[-1],
                              as_text=False)


class PyAIUtil(object):

    tf = tf





