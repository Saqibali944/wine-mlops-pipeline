install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

lint:
	python -m flake8 src/ tests/ --max-line-length=100

test:
	python -m pytest tests/ -v

train:
	python src/train.py

clean:
	rm -rf __pycache__ .pytest_cache mlruns
	find . -type f -name "*.pyc" -delete