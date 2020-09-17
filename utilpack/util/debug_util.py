# -*- coding: utf-8 -*-
"""
===============================================
debug module
===============================================

========== ====================================
========== ====================================
 Module     debug module
 Date       2019-03-26
 Author     heewinkim
========== ====================================

*Abstract*
    * debug 관련 유틸 모음

===============================================
"""
import time
import os

tic_ = 0
toc_ = 0

class PyDebugUtil(object):

    @staticmethod
    def tic():
        """
        타이머 시작

        :return:
        """
        global tic_
        tic_ = time.time()

    @staticmethod
    def toc(time_unit='ms', inplace_print=True,msg=''):
        """
        타이머 끝내기

        :param time_unit: ms, s 단위 측정
        :param inplace_print: True시 시간차를 즉시 출력
        :return: 시간 차이값
        """

        global tic_
        global toc_
        difftime = 0
        toc_ = time.time()

        if time_unit == 's':
            difftime = (toc_ - tic_)
        elif time_unit == 'ms':
            difftime = (toc_ - tic_) * 1000

        if inplace_print:
            if msg:
                msg = msg+' === '
            print("{}Difference Time : {}{}".format(msg,int(difftime), time_unit))

        return int(difftime)

    @staticmethod
    def timer_deco(func):
        def wrapper(*args, **kwargs):
            t1 = time.time()
            func(*args, **kwargs)
            t2 = time.time()
            dt = (t2 - t1) * 1000
            print('{} 함수가 실행되는데 걸린 시간: {:.2f}ms'.format(func.__name__, dt))

        return wrapper

    @staticmethod
    def memory_deco(func):
        import psutil
        def _get_process_memory():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss

        def wrapper(*args, **kwargs):
            mem_before = _get_process_memory()
            start = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
            mem_after = _get_process_memory()
            print("pid: {}\tfunc name: {}\tmemory before: {:,}\tafter: {:,}\tconsumed: {:,}\texec time: {}".format(
                os.getpid(),
                func.__name__,
                mem_before, mem_after, mem_after - mem_before,
                elapsed_time))
            return result

        return wrapper


if __name__ == '__main__':

    from time import sleep

    PyDebugUtil.tic()
    sleep(1)
    PyDebugUtil.toc()
    # Difference Time : 1001ms

    @PyDebugUtil.timer_deco
    def do_something():
        sleep(1)
    do_something()
    # do_something 함수가 실행되는데 걸린 시간: 1003.23ms

    @PyDebugUtil.memory_deco
    def do_something():
        sleep(1)
    do_something()
    # pid: 34597	func name: do_something	memory before: 69,550,080	after: 69,550,080	consumed: 0	exec time: 00:00:01