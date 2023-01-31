install:
	pip install --upgrade pip && pip install -r requirements.txt
lint:
	pylint --disable=R,C app.py
format-checks:
	pycodestyle --first *.py
format-fix:
	autopep8 --in-place --aggressive --aggressive *.py
test:
	python -m pytest -vv --cov=format test_app.py
all:
	install lint format-checks format-fix test