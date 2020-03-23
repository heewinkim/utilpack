pip install wheel
python setup.py bdist_wheel
pip install twine
twine upload dist/*