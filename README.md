## Hian common pakage  

### Note
    본 프로젝트는 일반적인 파이썬 프로젝트 진행시에 필요한 공통 모듈 패키지가 포함되어있습니다.
    keras_util의 경우 keras==2.2.0, tensorflow==1.8.0 을 직접 설치해야 합니다. 
---

### common.core
- config : HianConfig 클래스 제공, 설정파일 로드 기능을 제공
- error : HianError 에러클래스 제공, ERROR_TYPES의 에러 타입 제공
- image : HianImage 클래스 제공, 이미지 읽기, 및 핵심 처리기능 제공
- logger : HianLogger 클래스 제공, 로그관련 처리 기능 제공
- output : HianOutput 클래스 제공, output 관련기능, 에러에 대한 로그처리 기능 제공
- time : HianTime 클래스 제공, 시간 관련된 핵심 처리기능 제공

### common.util

- data_util : HianDataUtil 클래스 제공, 데이터 분석 및 처리 기능 제공
- debug_util : 버그에 필요한 모듈 제공
- image_util : HianImageUtil 클래스 제공, 이미지 분석관련 기능 제공
- keras_util :HianKearsUtil 케라스 학습 및 분석 기능 제공
- time_util : HianTimeUtil 클래스 제공, 시간 관련 분석 기능 제공
- train_util : HianTrainUtil 클래스 제공, 습 관련 데이터 전후처리 기능 제공

### common.framwork  

- hian_flask : Flask로 구현된 웹 API 클래스인 HianFlask 를 제공합니다.  


- EXAMPLE
        
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
        
        
        from framework.hian_flask import HianFlask
        
        example_api = ExampleApi()
        application = HianFlask('ex')
        
        
        @application.route('/hian/v1/example',methods=['POST','GET'])
        def request_api():
        
            application.output.set_default(['arg1','arg2'],-1)
            result = example_api.run()
            return application.output.return_output(result)
        
        if __name__ == '__main__':
            application.run('0.0.0.0',port=5000)    
        
