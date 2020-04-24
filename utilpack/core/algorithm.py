# -*- coding: utf-8 -*-
"""
===============================================
algorithm module
===============================================

========== ====================================
========== ====================================
 Module     algorithm module
 Date       2019-03-26
 Author     heewinkim
========== ====================================

*Abstract*
    * math 관련 정적메소드 제공

===============================================
"""
import numpy as np


class PyAlgorithm(object):

    @staticmethod
    def limit_minmax(x, min_=0, max_=None):
        return max(min_, x) if not max_ else min(max(min_, x), max_)

    @staticmethod
    def get_connected_components(pairs):
        """
        그래프적으로 연결된 노드들을 그룹화합니다.

        :param pairs: 한쌍의 노트이름의 리스트 eg.[ [1,2], [2,7], [0,3], [4,5], [5,7] ]
        :return: 연결된 그룹 리스트. eg.[[1, 2, 5, 7], [0, 3], [4, 5]]
        """

        if len(pairs)==0:
            return []

        set_list = [set(pairs[0])]

        for pair in pairs[1:]:
            matched = False
            for a_set in set_list:
                if pair[0] in a_set or pair[1] in a_set:
                    a_set.update(pair)
                    matched = True
                    break
            if not matched:
                set_list.append(set(pair))

        return [sorted(v) for v in set_list]

    @staticmethod
    def normalize(values,tolist=False):
        """
        합이 1이되도록 list를 normalize합니다.

        :param values: list
        :param tolist: True인경우 list 포맷으로 아닐경우 np.array포맷으로 반환됩니다
        :return: list
        """
        if type(values) != list or type(values) != np.array:
            values = list(values)

        if np.sum(values) == 0:
            if tolist:
                return list(values)
            else:
                return np.array(values)
        else:
            norm = np.linalg.norm(values, ord=1)

        if tolist:
            return list(values / norm)
        else:
            return values / norm


    @staticmethod
    def rank(values,startFrom=0,reverse=False,indices=False,tolist=False):
        """
        0부터 시작하하며 리스트의 순위를 얻습니다.

        :param values: list,any
        :param reverse: 참일시 값이 클수록 순위가 높습니다.
        :param indices: rank 에 따른 index를 반환합니다.
        :return: list
        """

        result = np.argsort(values)

        if reverse:
            result = result[::-1]

        if startFrom!=0:
            result = result + int(startFrom)

        if not indices:
            result = np.argsort(result)

        if tolist:
            result = list(result)

        return result

    @staticmethod
    def sortByValues(datas,values,reverse=False,sortFunc=None):
        """
        values 값을 key로하여 data를 정렬합니다.

        :param datas: data list to sorted
        :param values: values which using sort key
        :param sortFunc: key function of sort
        :return:
        """
        if not sortFunc:
            sortFunc = lambda v:v[1]

        if not len(datas)==len(values):
            raise ValueError("values must have same length with datas")

        return [d for d,v in sorted(list(zip(datas,values)),key=sortFunc,reverse=reverse)]

