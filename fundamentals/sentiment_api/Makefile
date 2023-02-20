install:
	pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt --no-cache-dir
lint:
	pylint --disable=R,C app.py
format-checks:
	pycodestyle --first *.py
format-fix:
	autopep8 --in-place --aggressive --aggressive *.py
tests:
	python -m pytest -vv --cov=format test_app.py
all:
	make install lint format-checks format-fix tests