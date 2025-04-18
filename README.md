# [utilpack](https://github.com/heewinkim/utilpack)
    본 프로젝트는 일반적인 파이썬 프로젝트 진행시에 필요한 유틸 모듈 패키지가 포함되어있습니다.

![](https://img.shields.io/badge/python-3.11.12-blue)


### 설치 방법 (How To Install)

```sh
pip3 install utilpack
```


### 사용 예제 (Example)

## common.core

<details>
<summary>PyAlgorithm</summary>
<p>

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

</p>
</details> 


<details>
<summary>PyConfig</summary>
<p>

- PyConfig 클래스 제공, 설정파일 로드 기능을 제공
```python
# ./py.conf 파일의 내용은 아래와 같이 정의되며 패키지루트경로/core 디렉토리에 포함됩니다.
# 해당 파일은 PyFlask,PyLogger 를 사용시 useFileHandler,td_log,slack_notify 등 사용할때 정의되는 값들을 포함합니다. 
# 예를들어 flask = PyFlask('app_name',useFileHandler=True)과 같이 파일에 로그하는 경우,
# PyConfig의 ['LOG']['PATH'] 값인 /opt/py/log 디렉토리 밑의 app_name 디렉토리에 API관련된 로그파일이 기록되게 됩니다.  
```

</p>
</details> 

<details>
<summary>PyCrypto</summary>
<p>

- PyCrypto 클래스 제공, AES256 등 암복호화 제공
```python
from utilpack.core import PyCrypto

crypto_obj = PyCrypto.AES256(key='key',block_size=32)

encrypt_data = crypto_obj.encrypt('example_data')
print(encrypt_data)  # cqw06setVz83Sy4aMpOjFeqbOKNfmRFOaIVqtYCogvFyXAhzbPrnoY+khmUfn+Q4

decrypt_data = crypto_obj.decrypt(encrypt_data)
print(decrypt_data)  # example_data
```

</p>
</details> 

<details>
<summary>PyData</summary>
<p>

- PyData 클래스 제공, 압축관련 및 데이터 전후처리 제공
```python
from utilpack.core import PyData
# 데이터를 압축합니다.
compress_data = PyData.zip.compress('string data',filename='filename')
print(compress_data)  # b'PK\x03\x04\x14...

# 압축된 데이터를 읽습니다.
decompress = data = PyData.zip.decompress(compress_data,filename='filename')
print(decompress)  # b'string data'

# 데이터를 json 형태로 저장합니다.
PyData.save_json({'a':1,'b':2},'sample.json')

# json 데이터파일을 읽습니다.
data = PyData.load_json('sample.json')
print(data)  # {'a': 1, 'b': 2}

# 데이터를 pickle 형태로 저장합니다.
PyData.save_pickle({'a':1,'b':2},'sample.pkl')

# pickle 데이터파일을 읽습니다.
data = PyData.load_pickle('sample.pkl')
print(data)  # {'a': 1, 'b': 2}
```

</p>
</details> 

<details>
<summary>PyError</summary>
<p>

- PyError 에러클래스 제공, ERROR_TYPES의 에러 타입 제공
```python
from utilpack.core import PyError,ERROR_TYPES

def do_something():
    raise PyError(ERROR_TYPES.RUNTIME_ERROR,'example error')

try:
    do_something()

except Exception as e:
    print(e.err_type.name,hex(e.err_type.value))  # RUNTIME_ERROR 0x70003
    print(e)  # [RUNTIME_ERROR] example error
```

</p>
</details> 

<details>
<summary>PyImage</summary>
<p>

- PyImage 클래스 제공, 이미지 읽기, 및 핵심 처리기능 제공
```python
from utilpack.core import PyImage

# 이미지의 크기 비율을 계산합니다 ot = exif orientation value ( 사진 회전값 )
ratio = PyImage.calculate_sizerate(ot=1,w=800,h=600)
print(ratio)  # 1.3333333333333333

# 이미지를 바이트 형태로 읽습니다, filepath, url, bytes 타입들의 이미지데이터를 지원합니다.
bytes_data = PyImage.read_bytedata('https://homepages.cae.wisc.edu/~ece533/images/airplane.png','url')
print(bytes_data[:10])  # b'\x89PNG\r\n\x1a\n\x00\x00'

# 이미지 bytes_data를 opencv-python 패키지의 이미지형식인 ndarray로 변환합니다.
img_cv = PyImage.bytes2cv(bytes_data)

# img_cv -> bytes_data
bytes_data = PyImage.cv2bytes(img_cv)
print(bytes_data[:10])  # b'\xff\xd8\xff\xe0\x00\x10JFIF'

# 이미지를 사진회전값에 맞게 회전시킵니다. 이미지에 드로잉 작업을 할 경우 copy=True로 하여 참조가 아닌 복사가 이루어져야 합니다.
roatated_img = PyImage.rotate_image(img_cv,orientation=6,copy=False)

# cv2 -> base64-jpeg format, tostring이 참인경우, string 타입으로 최종 변환됩니다.
img_b64_str = PyImage.cv2base64(img_cv,tostring=True)
print(img_b64_str[:10])  # /9j/4AAQSk

# bytes_data -> base64-jpeg format, tostring이 참인경우, string 타입으로 최종 변환됩니다.
img_b64_str = PyImage.byte2base64(bytes_data,tostring=True)
print(img_b64_str[:10])  # /9j/4AAQSk

# 이미지 헤더를 읽어 사이즈와 포맷을 체크 합니다, 사이즈가 너무작거나 큰경우, 지원하지 않는 이미지 포맷인경우 ERROR_TYPES.IMAGE_FORMAT_ERROR 에러를 발생시킵니다.(PyError 참조)
PyImage.check_img_sz_fmt(bytes_data,min_size=(20,20),max_size=(10000,10000),allowed_extensions={'png','jpg','jpeg','bmp'})

# filepath, url, bytes 등의 이미지 소스로 부터 cv2,base64-jpg 의 이미지포맷의 변환 및 이미지 검수,회전 등의 일련의 과정을 진행합니다.
# 읽기 실패, 데이터오류 등 발생할 수 있는 에러들에 대해 IMAGE_FORMAT_ERROR,IMAGE_READ_ERROR 등을 발생시킵니다.(PyError 참조)
img_b64 = PyImage.preprocessing_image('https://homepages.cae.wisc.edu/~ece533/images/airplane.png','url',1,'cv2')
```

</p>
</details> 

<details>
<summary>PyLogger</summary>
<p>

- PyLogger 클래스 제공, 로그관련 처리 기능 제공
```python
from utilpack.core import PyLogger

# PyLogger는 singletone 디자인패턴으로 객체화 합니다.
logger = PyLogger(log_name='pylog',useFileHandler=True)
# below directories are created, you can change the ROOTPATH infomation on PyConfig 
# {PACKAGE_ROOTPATH}/pylog/info_log
# {PACKAGE_ROOTPATH}/pylog/error_log

logger.info('example_log')

# {PACKAGE_ROOTPATH}/pylog/info_log/pylog_info.log
# INFO	26928	20-09-17 15:29:48	None	None	None	None	None	example_log 
# In above log content, None means request Infos which activated when using PyFlask
```

</p>
</details> 

<details>
<summary>PyMaths</summary>
<p>

- PyMaths 클래스 제공, 수학관련 기능 제공
```python
from utilpack.core import PyMaths

# 두 x,y좌표간 각도 계산
degree = PyMaths.get_degree((0,0),(10,10),degree90=True)
print(degree)  # 45.0
```

</p>
</details> 

<details>
<summary>PyOutput</summary>
<p>

- PyOutput 클래스 제공, output 관련기능, 에러에 대한 로그처리 기능 제공
```python
from utilpack.core import PyOutput,PyError,ERROR_TYPES

# PyOutput는 singletone 디자인패턴으로 객체화 합니다.
# PyOutput은 PyFlask에서 사용되어 REST API 구현시 json output을 관리합니다.
output = PyOutput(top_keys=['statusCode','message'])

# 저장된 output.output 데이터를 가져오며, 자동적으로 저장된 output.output 데이터는 초기화 됩니다.
# 스택의 pop과 비슷하다고 보면 됩니다.
print(output.get_output())  # {"statusCode": null, "message": null}

# keys,default_value를 받아 키를 하나의 값으로 초기화 하거나, dict_obj 값을 받아 output 디폴트 값을 update 합니다
# PyOutput 객체를 새로 생성하지 않고 초기값을 변경하는 것과 같습니다.
output.set_default(dict_obj={'example':'value'})
print(output.output)  # {'example': 'value'}

# output 데이터를 할당합니다.
output.set_output({'data':1234})
print(output.output)  # {'example': 'value', 'data': 1234}

# 현재 API의 진행상황에 에러가 있음을 표시합니다. (statusCode, message 데이터로 표기)
# PyError와 호환됩니다.
output.set_error(PyError(ERROR_TYPES.IMAGE_READ_ERROR,'example error'))
print(output.output)  # {'example': 'value', 'data': 1234, 'statusCode': 200, 'message': '[IMAGE_READ_ERROR] example error'}

# 현재 API의 진행상황이 성공적임을 표시합니다. (statusCode, message 데이터로 표기)
output.set_success()

# set_output(data) -> set_success() -> reset() 의 일련과정을 한번에 처리하는 strategy 패턴과 같은 메서드이며 output 데이터를 반환합니다.
returned_data = output.return_output({'example':'data'})
print(returned_data)  # {"example": "data", "data": 1234, "statusCode": 200, "message": "success"}

# output.output인 데이터를 초기화 합니다.
output.reset()
```

</p>
</details> 

<details>
<summary>PyTime</summary>
<p>

- PyTime 클래스 제공, 시간 관련된 핵심 처리기능 제공
```python
from utilpack.core import PyTime

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

dt = PyTime.get_timeFromFilename('20180501_235020')
print(dt)  # 2018-05-01 23:50:20

```

</p>
</details> 

    
## common.framwork  

<details>
<summary>PyFlask</summary>
<p>

- Flask로 구현된 웹 API 클래스인 PyFlask 를 제공합니다.  
```python
   
class ExampleApi(object):

    def run(self):
        result = {'arg1':100,'arg2':200}
        return result

    
from utilpack.framework import PyFlask

example_api = ExampleApi()
application = PyFlask(
    api_name = 'ex', # log 디렉토리 이름, health check url 경로, flask objeect name 등으로 설정됩니다.
    td_log = False, # fluentd 가 설정되어 있을 시 py.conf 의 설정값을 통해 원격 로깅 지원 
    logFilter = None, # input, output 데이터에 대한 필터처리 가능
    slackNotify = True # py.conf 의 설정을 통해 에러 발생시 메세징 기능 
)

@application.route('/py/v1/example',methods=['POST','GET'])
def request_api():

    application.output.set_default(['arg1','arg2'],-1)
    result = example_api.run()
    return application.output.return_output(result)

if __name__ == '__main__':
    application.run('0.0.0.0',port=5000)    
    
```

</p>
</details> 


