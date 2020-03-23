# -*- coding: utf-8 -*-
"""
===============================================
maths module
===============================================

========== ====================================
========== ====================================
 Module     maths module
 Date       2019-03-26
 Author     heewinkim
========== ====================================

*Abstract*
    * math 관련 정적메소드 제공

===============================================
"""
import numpy as np


class PyMaths(object):

    @staticmethod
    def get_degree(pt1,pt2,degree90=True):
        # compute the angle between the eye centroids
        dY = pt2[1] - pt1[1]
        dX = pt2[0] - pt1[0]
        angle = np.abs(np.degrees(np.arctan2(dY, dX)))
        if degree90:
            angle = 180-angle if angle>90 else angle

        return angle
