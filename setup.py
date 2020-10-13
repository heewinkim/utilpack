from setuptools import setup, find_packages

# pip install wheel           # 빌드 툴
# pip install setuptools     # 패키징 툴
# pip install twine            # 패키지 업로드 툴
require_packages=[
    'Flask',
    'gunicorn',
    'requests',
    'matplotlib',
    'sklearn',
    'pandas',
    'tqdm',
    'Pillow',
    'opencv-python',
    'pymysql',
    'ipywidgets',
    'pycryptodome',
    'shapely',
    'pdfrw'
]

packages = list(open('requirements.txt').readlines())
setup(
    name='utilpack',
    version='1.7.6',
    author='HEESEUNG KIM',
    author_email='heewin.kim@gmail.com',
    description='Python Utils',
    long_description="""Python Utils\n
    본 프로젝트는 일반적인 파이썬 프로젝트 진행시에 필요한 유틸 모듈 패키지가 포함되어있습니다.\n
    자세한 사용법은 Git 페이지를 확인해주세요.
    """,
    long_description_content_type='text/markdown',
    license='MIT',
    url='https://github.com/heewinkim/utilpack',
    download_url='https://github.com/heewinkim/utilpack/archive/master.zip',

    packages=find_packages(),
    install_requires=open('requirements.txt').readlines(),
    package_data={'':['*']},
    python_requires='>=3',
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)