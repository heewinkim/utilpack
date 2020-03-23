## utilpack  


### Note
    본 프로젝트는 일반적인 파이썬 프로젝트 진행시에 필요한 유틸 모듈 패키지가 포함되어있습니다.
    
---

### common.core

- PyConfig 클래스 제공, 설정파일 로드 기능을 제공
- PyError 에러클래스 제공, ERROR_TYPES의 에러 타입 제공
- PyImage 클래스 제공, 이미지 읽기, 및 핵심 처리기능 제공
- PyLogger 클래스 제공, 로그관련 처리 기능 제공
- PyOutput 클래스 제공, output 관련기능, 에러에 대한 로그처리 기능 제공
- PyTime 클래스 제공, 시간 관련된 핵심 처리기능 제공
- PyAlgorithm 클래스 제공

   
    >>> EXMAPLE
    from utilpack.core import PyAlgorithm
    
    pair_list = [ [1,2], [2,7], [0,3], [4,5], [5,7] ]
    result = PyAlgorithm.get_connected_components(pari_list)
    print(result)


### common.util

- PyDataUtil 클래스 제공, 데이터 분석 및 처리 기능 제공
- 디버그에 필요한 모듈 제공
- PyImageUtil 클래스 제공, 이미지 분석관련 기능 제공
- PyTimeUtil 클래스 제공, 시간 관련 분석 기능 제공
- PyVisUtil 클래스 제공, 시각화 툴 제공


    >>> EXMAPLE
    from utilpack.util import PyDebugUtil
    
    PyDebugUtil.tic()
    PyDebugUtil.toc()
    
### common.framwork  

- Flask로 구현된 웹 API 클래스인 PyFlask 를 제공합니다.  


    >>> EXMAPLE
    
    # -*- coding: utf-8 -*-
    """
    본 코드는 API 작성에 대한 예시 입니다.
    예시 API로는 본 코드에 작성된
    ExampleApi 클래스를 사용합니다.
    실제 API 사용시에는 API 클래스를 import 하여 사용합니다.
    """
    
    
    class ExampleApi(object):
    
        def run(self):
            result = {'arg1':100,'arg2':200}
            return result
    
    
    from utilpack.framework import PyFlask
    
    example_api = ExampleApi()
    application = PyFlask('ex')
    
    
    @application.route('/py/v1/example',methods=['POST','GET'])
    def request_api():
    
        application.output.set_default(['arg1','arg2'],-1)
        result = example_api.run()
        return application.output.return_output(result)
    
    if __name__ == '__main__':
        application.run('0.0.0.0',port=5000)    
    
