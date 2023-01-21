install:
	pip install --upgrade pip && pip install -r ./introduction/python-scaffold/requirements.txt
lint:
	pylint --disable=R,C ./introduction/python-scaffold/hello.py
test:
	python -m pytest -vv --cov=hello ./introduction/python-scaffold/test_hello.py