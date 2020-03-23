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