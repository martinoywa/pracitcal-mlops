install:
	pip install --upgrade pip && pip install -r ./introduction/python-scaffold/requirements.txt
lint:
	pylint --disable=R,C ./introduction/python-scaffold/hello.py
format-checks:
	pycodestyle --first ./introduction/python-scaffold/*.py
format-fix:
	autopep8 --in-place --aggressive --aggressive ./introduction/python-scaffold/*.py
test:
	python -m pytest -vv --cov=hello ./introduction/python-scaffold/test_hello.py