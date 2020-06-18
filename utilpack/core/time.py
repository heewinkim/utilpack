# -*- coding: utf-8 -*-
"""
===============================================
time module
===============================================

========== ====================================
========== ====================================
 Module     time module
 Date       2019-03-26
 Author     heewinkim
 Comment    `관련문서링크 <call to heewinkim >`_
========== ====================================

*Abstract*
    * Time 클래스를 제공합니다.
    * 시간 관련한 처리 함수를 제공합니다.

===============================================
"""

from random import random
from random import seed
import numpy as np
from collections import deque
from datetime import datetime
import re
from .error import *


class PyTime(object):

    @staticmethod
    def check_date_inrange(date, date_range):
        try:
            if int(date_range[0][:4]) * 10000 + int(date_range[0][5:7]) * 100 + int(date_range[0][8:10]) <= int(
                    date[:4]) * 10000 + int(date[5:7]) * 100 + int(
                date[8:10]) <= int(date_range[1][:4]) * 10000 + int(date_range[1][5:7]) * 100 + int(
                    date_range[1][8:10]):
                return True
            else:
                return False
        except Exception:
            return False

    @staticmethod
    def str2datetime(row):
        """
        Sting 날짜 포맷을 시간으로 변환

        :param row: string 날짜 데이터
        :return: datetime 포맷 변환 된 날짜
        """
        try:
            p = re.compile('[-: ]')
            return datetime(*map(int, p.split(row)))
        except Exception:
            return None

    @staticmethod
    def check_datetime(time):
        """
        datetime 형식이 맞는 지확인
        :param row: 입력된 날짜 데이터(str)('YYYY-MM-DD HH:MM:SS')
        :return: 1 or None
        """
        if type(time)==datetime:
            time = str(datetime)

        if time is not None and len(time)==19 and PyTime.str2datetime(time):
            return True
        else:
            return False

    @staticmethod
    def get_difftime(srctime, dsttime):
        """
        format 형식에 따르는 string 데이터 타입의 두 srctime, dsttime에 대한 시간차이를 초단위로 반환합니다.
        시간순으로 srctime이 빠르고 dsttime이 느린시간(src = 20180000, dst = 20181212) 입니다.
        :param srctime: string
        :param dsttime: string
        :param format: datetime클래스의 포맷
        :return: second which is time of difference between two times
        """
        try:
            return (PyTime.str2datetime(dsttime) - PyTime.str2datetime(srctime)).total_seconds()
        except Exception:
            raise PyError(ERROR_TYPES.PREPROCESSING_ERROR,'Invalid date format ("YYYY-mm-dd")')

    @staticmethod
    def get_diffday(srctime, dsttime):
        """
        format 형식에 따르는 string 데이터 타입의 두 srctime, dsttime에 대한 시간차이를 일단위로 반환합니다.
        시간순으로 srctime이 빠르고 dsttime이 느린시간(src = 20180000, dst = 20181212) 입니다.
        :param srctime: string
        :param dsttime: string
        :param format: datetime클래스의 포맷
        :return: second which is day of difference between two times
        """
        try:
            srctime = PyTime.str2datetime(srctime)
            dsttime = PyTime.str2datetime(dsttime)

            srctime = datetime(srctime.year, srctime.month, srctime.day, 0, 0, 0)
            dsttime = datetime(dsttime.year, dsttime.month, dsttime.day, 0, 0, 0)

            return (dsttime-srctime).total_seconds()
        except Exception:
            raise PyError(ERROR_TYPES.PREPROCESSING_ERROR,'Invalid date format ("YYYY-mm-dd")')

    @staticmethod
    def get_differential_times(images,time_type='exifDate'):
        """
        클러스터간의 시간미분값의 리스트를 반환
        원하는 타임 데이터의 리스트를 받는다
        각 리스트의 순서대로 image(object)의 값 존재유무를 판단하여 그 값을 시간데이터로 저장한다
        예를들어 exifDate로만 비교하고 싶다면 ['exifDate']를 넘기고
        exifDate로 하지만 없는경우 sysDate로 하고 싶다면 ['exifDate','sysDate'] 와 같은 방식이다
        단 각 이미지는 제공된 time_type 리스트중 하나라도 값을 가져야한다.

        :param images: image list, exifDate 키를 포함하는 딕셔너리 형태의 image
        :param time_type: images 한 원소에서 시간값을 갖는 키
        :return: list
        """
        try:
            differential_times = []
            if not len(images) >= 1:
                return differential_times

            previous_tail_time = images[0][time_type]

            for image in images:
                head_time = image[time_type]

                # 초단위로 미분값을 저장합니다.
                diff_time = abs(PyTime.get_difftime(previous_tail_time, head_time))

                differential_times.append(diff_time)
                previous_tail_time = head_time

        except Exception:
            raise PyError(ERROR_TYPES.PREPROCESSING_ERROR,'No {} data in some images'.format(time_type))

        return differential_times

    @staticmethod
    def get_mean_time(date_list):
        """
        :param date_list: YYYY:MM:DD HH:MM:SS 형식의 날짜리스트( format인 :, ,- 등은 상관없음)
        :return: 평균 datetime 포맷 변환 된 날짜
        """
        date_list = [PyTime.str2datetime(date) for date in date_list if PyTime.check_datetime(date)]
        if len(date_list)==0:
            return None
        else:
            average_time= datetime.fromtimestamp(sum(map(datetime.timestamp, date_list)) / len(date_list))
            return average_time

    @staticmethod
    def _grouping(images,min,max,time_type,differential_times_=None):
        """
        각 domain의 이미지가 포토북의 최소 최대 (self.cut_range) 값에 해당되도록
        시간미분 값을 기준으로 재귀적으로 분할합니다.

        :param key: 분할하고자하는 지역의 key
        :param images: 분할하고자 하는 지역의 이미지
        :param differential_times_: images에 대한 시간 차이 값 리스트
        :return: 분할된 clusters
        :raise AssertionError: min*2<=max
        """

        groups = []

        assert min * 2 <= max

        # 최소 2분할의 전제조건 체크
        if len(images)<min*2:
            return [images]

        # 시간 분할
        if differential_times_:
            differential_times = differential_times_
        else:
            differential_times = PyTime.get_differential_times(images,time_type)
        if not differential_times:
            return groups

        while True:

            # 모든 요소값이 같은 경우
            if len(set(differential_times)) == 1:
                differential_times = [random() for _ in range(len(differential_times))]

            # 가장 큰 시간미분 값을 가진 인덱스
            max_idx = differential_times.index(np.max(differential_times))

            if len(images[:max_idx]) >= min and len(images[max_idx:]) >= min:

                # 잘린 앞부분이 포토북 범위에 들어간다면
                if len(images[:max_idx]) <= max:
                    groups.append(images[:max_idx])

                # 잘린 앞부분이 포토북 범위보다 크다면(재귀)
                elif len(images[:max_idx]) > max:
                    groups_ = PyTime._grouping(images[:max_idx], min, max,time_type,differential_times[:max_idx])
                    groups.extend(groups_)

                # 잘린 뒷부분이 포토북 범위에 들어간다면
                if len(images[max_idx:]) <= max:
                    groups.append(images[max_idx:])

                # 잘린 뒷부분이 포토북 범위보다 크다면(재귀)
                elif len(images[max_idx:]) > max:
                    groups_ = PyTime._grouping(images[max_idx:], min, max,time_type,differential_times[max_idx:])
                    groups.extend(groups_)
                break
            else:
                differential_times[max_idx] = 0

        return groups

    @staticmethod
    def _grouping_postprocessing(groups,max):

        result = []

        # 너무 작게 나누어진 그룹 결과 다시 병합
        dq_groups = deque(groups)
        while dq_groups:
            curr_group = dq_groups.popleft()
            result.append(curr_group)
            if not dq_groups:
                break
            next_group = dq_groups.popleft()
            if len(curr_group) + len(next_group) <= max:
                result[-1] = result[-1] + next_group
            else:
                result.append(next_group)

        return result

    @staticmethod
    def grouping_bytimediff(images,min,max,time_type='exifDate',after_merge=True,seed_value=None,sort=False):
        """
        스냅스 이미지객체 리스트를 받아
        이미지개수범위를 min,max에 맞게 분할합니다.
        분할기준은 시간미분값을 기준으로 합니다.

        autor : heewinkim

        :param images: 이미지 배열
        :param min: 분할된 그룹의 최소 이미지 개수
        :param max: 분할된 그룹의 최대 이미지 개수
        :param time_type: 시간 미분값으로 사용될 정보, exif,sys 둘중 하나 혹은 둘다, 둘다인 경우 exif우선으로 미분값을 계산
        :param after_merge: 시간미분정보으로 우선적인 그룹핑 후에 앞뒤 그룹(시간이 가까운) 끼리 합쳤을때 min,max범위에 들어간다면 병합 (시간정보로 나누지만 min,max의 기준이 더 우선권인 경우)
        :param seed_value: 시간값에 의한 그룹핑이 가능하지 않을경우 랜덤요소가 적용되며 그에대한 시드값
        :return: list, 이미지배열(그룹)의 배열
        """

        if seed_value:
            seed(seed_value)
        if sort:
            images = sorted(images,key=lambda v:v[time_type])

        groups = PyTime._grouping(images,min,max,time_type)
        if after_merge:
            groups = PyTime._grouping_postprocessing(groups,max)
        return groups

    @staticmethod
    def get_period(dates):
        """
        기간을 구합니다.
        유일한 기간 하나만 있는경우 [ unique date, unique date ] 와 같이 같은 날짜의 기간으로 표현됩니다.
        데이터 충분치 않은거나(0개 인 경우) 잘못된 날짜데이터로만 이루어진경우 None으로 표현됩니다.

        예시
        [ None, None ]


        :param dates: date(str,'%Y-%m-%d %H:%M:%S) list
        :return: [ pastest date, lastest date ]
        """
        time_period = [None, None]

        dates = sorted([date for date in dates if date])

        for date in dates:
            if PyTime.check_datetime(date):
                st_date = PyTime.str2datetime(date)
                time_period[0] = '{:04}-{:02}-{:02}'.format(st_date.year, st_date.month, st_date.day)
                break
        for date in reversed(dates):
            if PyTime.check_datetime(date):
                ed_date = PyTime.str2datetime(date)
                time_period[1] = '{:04}-{:02}-{:02}'.format(ed_date.year, ed_date.month, ed_date.day)
                break

        return time_period

