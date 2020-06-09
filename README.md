## utilpack
    본 프로젝트는 일반적인 파이썬 프로젝트 진행시에 필요한 유틸 모듈 패키지가 포함되어있습니다.

![](https://img.shields.io/badge/python-3.6.1-blue)


### 설치 방법

```sh
pip3 install utilpack
```

### 사용 예제

#### common.core

- PyAlgorithm 클래스 제공
```python
from utilpack.core import PyAlgorithm

pair_list = [ [1,2], [2,7], [0,3], [4,5], [5,7] ]

# 연결된 컴포넌트 반환
result = PyAlgorithm.get_connected_components(pair_list)
print(result)

```
- PyConfig 클래스 제공, 설정파일 로드 기능을 제공
```python

```
- PyData 클래스 제공, 압축관련 및 데이터 전후처리 제공
```python

```
- PyError 에러클래스 제공, ERROR_TYPES의 에러 타입 제공
```python

```
- PyImage 클래스 제공, 이미지 읽기, 및 핵심 처리기능 제공
```python

```
- PyLogger 클래스 제공, 로그관련 처리 기능 제공
```python

```
- PyMaths 클래스 제공, 수학관련 기능 제공
```python

```
- PyOutput 클래스 제공, output 관련기능, 에러에 대한 로그처리 기능 제공
```python

```
- PyTime 클래스 제공, 시간 관련된 핵심 처리기능 제공
```python

```

   


#### common.util

- PyDataUtil 클래스 제공, 데이터 분석 및 처리 기능 제공
```python

```
- PyDebugUtil 디버그에 필요한 모듈 제공
```python
from utilpack.util import PyDebugUtil

PyDebugUtil.tic()
PyDebugUtil.toc()
```
- PyFaceUtil 얼굴 이미지 관련 유틸 제공
```python

```
- PyImageUtil 클래스 제공, 이미지 분석관련 기능 제공
```python

```
- PyTimeUtil 클래스 제공, 시간 관련 분석 기능 제공
```python

```
- PyVisUtil 클래스 제공, 시각화 툴 제공
```python

```
- PyUI 클래스 제공, 주피터 UI 작성 유틸 제공
```python

``` 
    
#### common.framwork  

- Flask로 구현된 웹 API 클래스인 PyFlask 를 제공합니다.  

```python
   
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
    
```