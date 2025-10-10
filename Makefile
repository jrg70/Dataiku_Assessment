# =Makefile=
VENV = .venv

activate:
	source $(VENV)/bin/activate

run:
	python src/run.py

train:
	python src/train.py

evaluate:
	python src/evaluate_models.py

notebook:
	jupyter notebook notebooks/eda.ipynb

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf results/*.csv models/*.joblib

install:
	pip install -r requirements.txt
