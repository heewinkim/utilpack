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
import random
from shapely.geometry import box


class PyAlgorithm(object):

    @staticmethod
    def unionRects(rect_list):
        """
        find intersection of rects

        :param rect_list: [ (minx, miny, maxx, maxy), (minx, miny, maxx, maxy), ... ]
        :return: box object (shapely.geometry)
        """
        # make some rectangles (for demonstration purposes and intersect with each other)
        box_list = [ box(*rect) for rect in rect_list]
        union = box_list[0]

        # find intersection of rectangles (probably a more elegant way to do this)
        for box_ in box_list[1:]:
            union = union.union(box_)
        if union.area == box_list[0].area:
            if PyAlgorithm.intersectionRects(rect_list).area:
                return union.bounds
            else:
                return box(0,0,0,0)
        else:
            return union.bounds

    @staticmethod
    def intersectionRects(rect_list):
        """
        find intersection of rects

        :param rect_list: [ (minx, miny, maxx, maxy), (minx, miny, maxx, maxy), ... ]
        :return: box object (shapely.geometry)
        """
        # make some rectangles (for demonstration purposes and intersect with each other)
        box_list = [box(*rect) for rect in rect_list]
        intersection = box_list[0]

        # find intersection of rectangles (probably a more elegant way to do this)
        for box_ in box_list[1:]:
            intersection = intersection.intersection(box_)

        if intersection.area == box_list[0].area:
            if intersection.area <= PyAlgorithm.unionRects(rect_list).area:
                return intersection.bounds
            else:
                return box(0,0,0,0)
        else:
            return intersection.bounds

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

        if len(values)==0:
            return [] if tolist else np.array([])

        if reverse:
            values = np.max(values)-values

        result = np.argsort(values)

        if not indices:
            result = np.argsort(result)

        if startFrom!=0:
            result = result + int(startFrom)

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

    @staticmethod
    def sample_minimal_redundancy(arr, k, seed=None):
        """
        최소한의 중복을 허용하는 선에서 arr 배열중에서 k개의 샘플을 뽑습니다.

        :param arr: 배열
        :param k: 선택할 개수
        :param seed: 랜덤 시드
        :return: list
        """

        result = []
        if seed is not None:
            random.seed(seed)

        # 빈 배열일 경우 빈 리스트를 리턴합니다.
        if len(arr) == 0:
            return result

        for _ in range(k // len(arr)):
            result.extend(random.sample(arr, k=len(arr)))
        if k % len(arr) != 0:
            result.extend(random.sample(arr, k=k % len(arr)))
        return result


if __name__ == '__main__':

    rst = PyAlgorithm.intersectionRects([[0, 0, 20, 20], [10, 10, 30, 30]])
    print(rst)  # (10.0, 10.0, 20.0, 20.0)
    rst = PyAlgorithm.unionRects([[0, 0, 20, 20], [10, 10, 30, 30]])
    print(rst)  # (0.0, 0.0, 30.0, 30.0)
    rst = PyAlgorithm.limit_minmax(256,0,255)
    print(rst)  # 255
    rst = PyAlgorithm.get_connected_components([ [1,2], [2,7], [0,3], [4,5], [5,7] ])
    print(rst)  # [[1, 2, 5, 7], [0, 3], [4, 5]]
    rst = PyAlgorithm.rank([1,10,3,15,40],startFrom=1,indices=False,reverse=True)
    print(rst)  # [5 3 4 2 1]
    rst = PyAlgorithm.sortByValues(['a','b','c','d'],[4,3,1,2])
    print(rst)  # ['c', 'd', 'b', 'a']
    rst = PyAlgorithm.sample_minimal_redundancy([1,2,3,4,5],k=7,seed='random_seed')
    print(rst)  # [3, 2, 4, 5, 1, 4, 1]
