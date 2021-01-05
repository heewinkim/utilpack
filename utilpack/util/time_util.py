# -*- coding: utf-8 -*-
"""
===============================================
time_util module
===============================================

========== ====================================
========== ====================================
 Module     time_util module
 Date       2019-03-26
 Author     heewinkim
========== ====================================

*Abstract*
    * 시간 관련 유틸 모음

===============================================
"""
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import signal
from utilpack.core import PyTime


class PyTimeUtil(object):

    @staticmethod
    def draw_timeline(date_list, figsize=(20, 3), hour_interval=None):
        """
        타임 라인을 그립니다


        :param date_list: YYYY:MM:DD HH:MM:SS 형식의 날짜리스트( format인 :, ,- 등은 상관없음)
        :param figsize: 출력 사이즈
        :param hour_interval: 시간단위의 x축 ticks 정밀도
        :return: None
        """

        date_format = '%Y:%m:%d %H:%M:%S'

        date_list = ['{}:{}:{} {}:{}:{}'.format(date[:4], date[5:7], date[8:10], date[11:13], date[14:16], date[17:19])
                     for date in date_list]
        dates = [datetime.strptime(date, date_format) for date in date_list]

        fig, ax = plt.subplots(figsize=figsize)

        start = min(dates) - timedelta(days=1)
        stop = max(dates) + timedelta(days=1)

        if hour_interval is not None:
            sec_interval = hour_interval * 60 * 60
        else:
            sec_interval = int((stop - start).total_seconds() / 10)

        ax.plot((start, stop), (0, 0))
        ax.plot((start, start), (0, 10), alpha=0)

        for idate in dates:
            ax.plot((idate, idate), (0, 4), c='r', alpha=.5)

        ax.set_title('cluster timeline', color='#808080')
        ax.get_xaxis().set_major_locator(mdates.SecondLocator(interval=sec_interval))
        ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%m %d %Hh"))

        fig.autofmt_xdate()
        plt.setp((ax.get_yticklabels() + ax.get_yticklines() + list(ax.spines.values())), visible=False)
        ax.tick_params(axis='x', colors='gray')
        ax.set_facecolor('#FFFFFF00')
        plt.show()

    @staticmethod
    def seperate_data_bytime(data, unit='year',time_types=['exifDate']):
        """
        data를 unit단위에 맞게 나눕니다.

        eg. data = [
            { "value": 123,"exifDate": "2019-12-01 12:21:32" },
            { "value": 127,"exifDate": "2020-12-01 12:21:35" },
            { "value": 120,"exifDate": "2021-12-01 12:21:33" },
            ...
        ]
        output = {
            "2019": [{ "value": 123,"exifDate": "2019-12-01 12:21:32" }]
            "2020": [{ "value": 127,"exifDate": "2020-12-01 12:21:35" }]
            "2021": [{ "value": 120,"exifDate": "2021-12-01 12:21:33" }]
        }


        :param data: dict의 리스트 형태이며, time_types 들 중 하나의 필드를 가지고 있어야합니다.
        :param unit: 'year','month','day' 단위를 지원합니다.
        :param time_types: 시간값을 가진 속성이름 입니다. exifDate,sysDate등 리스트 형태이며 리스트 순서대로 우선권을 가집니다.
        :return:
        """
        result = {}
        for elem in data:
            time_type=''
            for t in time_types:
                if elem.get(t) and PyTime.check_datetime(elem[t]):
                    time_type = t
                    break

            if not time_type:
                return result

            if elem.get(time_type) and PyTime.check_datetime(elem[time_type]):
                date = PyTime.str2datetime(elem[time_type])
                if unit == 'year':
                    result[date.year] = result.get(date.year, []) + [elem]
                elif unit == 'month':
                    result[date.month] = result.get(date.month, []) + [elem]
                elif unit == 'day':
                    result[date.day] = result.get(date.day, []) + [elem]

        return result

# USE EXAMPLE
# with timeout(seconds=3):
#   do_something()

class Timeout:

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


if __name__ == '__main__':

    # 타임라인을 그립니다.
    PyTimeUtil.draw_timeline(['2020-09-07 12:11:10','2020-09-08 12:11:10','2020-09-10 12:11:10'])

    # 데이터를 시간단위로 나눕니다
    rst = PyTimeUtil.seperate_data_bytime([
            { "value": 123,"exifDate": "2019-12-01 12:21:32" },
            { "value": 127,"exifDate": "2020-12-01 12:21:35" },
            { "value": 120,"exifDate": "2021-12-01 12:21:33" },
    ],unit='year',time_types=['exifDate'])
    print(rst)