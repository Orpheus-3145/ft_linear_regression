VENV = .venv
INPUT ?= 100000
LEARNING_RATE ?= 0.01

$(VENV):
	@poetry install

all: build

build: $(VENV)

train: build
	@poetry run python src/train_model.py --learning_rate $(LEARNING_RATE)

train_range: build
	@clear
	@poetry run python src/train_model_range.py

estimate: build
	@poetry run python src/estimate_value.py --input $(INPUT)
