import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

def get_requirements():
    with open('requirements.txt') as f:
        lines = f.readlines()
        packages=[]
        for line in lines:
            line = line.strip()
            if line and '#' not in line:
                packages.append(line)
    return packages

a = get_requirements()

setuptools.setup(
    name="hian",
    version="0.0.1",
    author="hian",
    author_email="heewin.kim@gmail.com",
    description="python3 development tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/heewinkim/hian.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache 2.0",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.6.1'
    ],
    install_requires=get_requirements(),
    python_requires='>=3.6.1',
    zip_safe=False
)