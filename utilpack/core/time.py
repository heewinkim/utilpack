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
        """
        데이터 포맷은 모두 string, 'YYYY-mm-dd HH:MM:SS' 이어야 합니다.

        :param date: string datetime format
        :param date_range: list,[st_date,end_date]
        :return:
        """
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
        :param row: 입력된 날짜 데이터간(str)('YYYY-MM-DD HH:MM:SS')
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
        format 형식에 따르는 string 데이터 타입의 두 srctime, dsttime에 대한 일 단위차이를 초단위로 반환합니다.
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
    def get_differential_times(obj_list,time_type='exifDate'):
        """
        time_list의 시간 차이 리스트를 구합니다. 처음 시간은 앞의 시간이 없으므로 시간차를 0으로 할당합니다.

        :param obj_list: time_type필드가 포함된 객체 배열
        :param precision: 'sec','day' 지원
        :return: list
        """
        try:
            differential_times = []
            if not len(obj_list) >= 1:
                return differential_times

            previous_tail_time = obj_list[0][time_type]

            for obj in obj_list:
                head_time = obj[time_type]

                # 초단위로 미분값을 저장합니다.
                diff_time = abs(PyTime.get_difftime(previous_tail_time, head_time))

                differential_times.append(diff_time)
                previous_tail_time = head_time

        except Exception:
            if time_type=='sysDate':
                raise PyError(ERROR_TYPES.PARAMETER_ERROR, 'Invalid sysDate data in some obj_list'.format(time_type))
            else:
                raise PyError(ERROR_TYPES.PREPROCESSING_ERROR,'Invalid exifDate data in some obj_list'.format(time_type))

        return differential_times

    @staticmethod
    def get_mean_time(date_list):
        """
        주어진 date 리스트 중 평균 날짜를 구합니다.

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
    def _grouping(obj_list,min,max,time_type,differential_times_=None):
        """
        시간미분 값을 기준으로 재귀적으로 분할합니다.

        :param obj_list: 분할하고자 하는 obj 리스트
        :param min: 분할된 파편의 최소 길이
        :param max: 분할된 파편의 최대 길이
        :param time_type: obj_list의 원소에서 기준 시간값을 나타내 필드이름
        :param differential_times_: obj_list 대한 시간 차이 값 리스트
        :raise AssertionError: min*2<=max
        """

        groups = []

        assert min * 2 <= max

        # 최소 2분할의 전제조건 체크
        if len(obj_list)<min*2:
            return [obj_list]

        # 시간 분할
        if differential_times_:
            differential_times = differential_times_
        else:
            differential_times = PyTime.get_differential_times(obj_list,time_type)
        if not differential_times:
            return groups

        while True:

            # 모든 요소값이 같은 경우
            if len(set(differential_times)) == 1:
                differential_times = [random() for _ in range(len(differential_times))]

            # 가장 큰 시간미분 값을 가진 인덱스
            max_idx = differential_times.index(np.max(differential_times))

            if len(obj_list[:max_idx]) >= min and len(obj_list[max_idx:]) >= min:

                # 잘린 앞부분이 범위에 들어간다면
                if len(obj_list[:max_idx]) <= max:
                    groups.append(obj_list[:max_idx])

                # 잘린 앞부분이 범위보다 크다면(재귀)
                elif len(obj_list[:max_idx]) > max:
                    groups_ = PyTime._grouping(obj_list[:max_idx], min, max,time_type,differential_times[:max_idx])
                    groups.extend(groups_)

                # 잘린 뒷부분이 범위에 들어간다면
                if len(obj_list[max_idx:]) <= max:
                    groups.append(obj_list[max_idx:])

                # 잘린 뒷부분이 범위보다 크다면(재귀)
                elif len(obj_list[max_idx:]) > max:
                    groups_ = PyTime._grouping(obj_list[max_idx:], min, max,time_type,differential_times[max_idx:])
                    groups.extend(groups_)
                break
            else:
                differential_times[max_idx] = 0

        return groups

    @staticmethod
    def _grouping_postprocessing(groups,max):
        """
        너무 작게 나누어진 그룹 결과 다시 병합

        :param groups: 나누어진 그룹들
        :param max: 병합시 그룹길이의 최대길이
        :return: 병합된 그룹, list
        """

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
    def grouping_bytimediff(obj_list,min,max,time_type='exifDate',after_merge=True,seed_value=None,sort=False):
        """
        객체 리스트를 받아 이미지개수범위를 min,max에 맞게 분할합니다.
        분할기준은 시간미분값을 기준으로 하며, obj_list의 원소모두에 time_type 필드가 필수로 있어야합니다.
        Eg. obj_list = [{'index':0,'exifDate':2020-09-09 12:00:00'},{'index':1,'exifDate':2020-09-09 12:00:01'},...]

        ahutor : heewinkim

        :param obj_list: time_type필드가 포함된 객체 배열
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
            obj_list = sorted(obj_list,key=lambda v:v[time_type])

        if min==max:
            groups = sorted([list(v) for v in np.array_split(np.array(obj_list), len(obj_list)//min)],key=lambda v: v[0][time_type])
            if all([len(v)==min for v in groups]):
                return groups
            else:
                raise PyError(ERROR_TYPES.PREPROCESSING_ERROR,"Can't split obj_list({}) by {}".format(len(obj_list),min))
        else:
            groups = PyTime._grouping(obj_list,min,max,time_type)
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


if __name__ == '__main__':


    # 시간이 해당 기간 안에 포함되는지 체크합니다.
    ret = PyTime.check_date_inrange('2020-09-09 12:11:10',['2020-09-08 12:11:10','2020-09-10 12:11:10'])
    print(ret)  # True

    # 올바른 시간 데이터 str, ('YYYY-mm-dd HH:MM:SS') 인지 확인합니다.
    ret = PyTime.check_datetime('2020-09-09 12:11:10')
    print(ret)  # True

    # Sting 날짜 포맷을 시간으로 변환
    datetime_data = PyTime.str2datetime('2020-09-09 12:11:10')
    print(datetime_data)  # 2020-09-09 12:11:10

    # string datetime(YYYY-mm-dd HH:MM:SS)의 두 srctime, dsttime에 대한 시간차이를 초단위로 반환합니다.
    difftime_seconds = PyTime.get_difftime('2020-09-09 12:11:10','2020-09-10 04:00:00')
    print(difftime_seconds)  # 56930.0

    # string datetime(YYYY-mm-dd HH:MM:SS)의 두 srctime, dsttime에 대한 일 단위차이를 초단위로 반환합니다.
    diff_days = PyTime.get_diffday('2020-09-09 12:11:10','2020-09-10 04:00:00')
    print(diff_days)  # 86400.0

    # time_list의 시간 차이 리스트를 구합니다. 객체 리스트를 받으며 각 객체 리스트는 time_type을 포함해야 합니다.
    difference_timedays = PyTime.get_differential_times(
        [
            {'exifDate': '2020-09-07 12:11:10'},
            {'exifDate': '2020-09-08 12:11:10'},
            {'exifDate': '2020-09-10 12:11:10'}
        ],
        time_type='exifDate')
    print(difference_timedays)  # [0.0, 86400.0, 172800.0]

    # 주어진 date 리스트 중 평균 날짜를 구합니다.
    mean_data = PyTime.get_mean_time(['2020-09-07 12:11:10','2020-09-08 12:11:10','2020-09-10 12:11:10'])
    print(mean_data)  # 2020-09-08 20:11:10

    # 객체 리스트를 시간기준으로 적절할게 분할합니다.
    groups = PyTime.grouping_bytimediff(
        obj_list=[
            {'exifDate': '2020-09-05 12:11:10','data':1},
            {'exifDate': '2020-09-06 12:11:10','data':2},
            {'exifDate': '2020-09-07 12:11:10','data':3},
            {'exifDate': '2020-09-08 12:11:10','data':4},
            {'exifDate': '2020-09-09 12:11:10','data':5},
        ],
        min=2,max=4,time_type='exifDate',after_merge=True,seed_value=1234,sort=False
    )
    print(groups)
    # [
    #   [
    #     {'exifDate': '2020-09-05 12:11:10', 'data': 1},
    #     {'exifDate': '2020-09-06 12:11:10', 'data': 2}
    #   ],
    #   [
    #     {'exifDate': '2020-09-07 12:11:10', 'data': 3},
    #     {'exifDate': '2020-09-08 12:11:10', 'data': 4},
    #     {'exifDate': '2020-09-09 12:11:10', 'data': 5}
    #   ]
    # ]

    # date 리스트를 받아 [가장 오래된 날짜, 가장 최근 날짜] 를 얻습니다.
    period = PyTime.get_period(['2020-09-07 12:11:10','2020-09-08 12:11:10','2020-09-10 12:11:10'])
    print(period)  # ['2020-09-07', '2020-09-10']
