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
    def normalize(v):
        if type(v) != list or type(v) != np.array:
            v = list(v)
        if np.sum(v) == 0:
            return v
        else:
            norm = np.linalg.norm(v, ord=1)
        return list(v / norm)
