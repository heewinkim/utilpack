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
]

packages = list(open('requirements.txt').readlines())
setup(
    name='utilpack',
    version='1.1.4',
    author='HEESEUNG KIM',
    author_email='heewin.kim@gmail.com',
    description='Python Utils',
    long_description=open('README.md').read(),
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