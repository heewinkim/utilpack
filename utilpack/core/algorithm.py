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
    def normalize(values):
        """
        합이 1이되도록 list를 normalize합니다.

        :param values: list
        :return: list
        """
        if type(values) != list or type(values) != np.array:
            values = list(values)
        if np.sum(values) == 0:
            return values
        else:
            norm = np.linalg.norm(values, ord=1)
        return list(values / norm)

    @staticmethod
    def rank(values,reverse=False):
        """
        0부터 시작하하며 리스트의 순위를 얻습니다.

        :param values: list,any
        :param reverse: 참일시 값이 클수록 순위가 높습니다.
        :return: list
        """

        if reverse:
            return list(np.argsort(np.argsort(values)[::-1]))
        else:
            return list(np.argsort(np.argsort(values)))

if __name__ == '__main__':
    print(PyAlgorithm.get_connected_components([ [1,2], [2,7], [0,3], [4,5], [5,7] ]))