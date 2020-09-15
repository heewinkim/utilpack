# [utilpack](https://github.com/heewinkim/utilpack)
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
# 사각형의 교집합을 구합니다.
rst = PyAlgorithm.intersectionRects([[0, 0, 20, 20], [10, 10, 30, 30]])
print(rst)  # (10.0, 10.0, 20.0, 20.0)
# 사각형의 합집합을 구합니다.
rst = PyAlgorithm.unionRects([[0, 0, 20, 20], [10, 10, 30, 30]])
print(rst)  # (0.0, 0.0, 30.0, 30.0)
# 값의 최소 최대값을 적용합니다.
rst = PyAlgorithm.limit_minmax(256,0,255)
print(rst)  # 255
# 각 노드의 엣지관계 리스트를 받아 연결 컴포넌트를 구합니다.
rst = PyAlgorithm.get_connected_components([ [1,2], [2,7], [0,3], [4,5], [5,7] ])
print(rst)  # [[1, 2, 5, 7], [0, 3], [4, 5]]
# 값의 랭크를 매깁니다.
rst = PyAlgorithm.rank([1,10,3,15,40],startFrom=1,indices=False,reverse=True)
print(rst)  # [5 3 4 2 1]
# values 의 정렬 값으로 data를 정렬합니다.
rst = PyAlgorithm.sortByValues(['a','b','c','d'],[4,3,1,2])
print(rst)  # ['c', 'd', 'b', 'a']
# 최소한의 중복을 허용하는 선에서 arr 배열중에서 k개의 샘플을 뽑습니다.
rst = PyAlgorithm.sample_minimal_redundancy([1,2,3,4,5],k=7,seed='random_seed')
print(rst)  # [3, 2, 4, 5, 1, 4, 1]
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