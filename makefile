VENV_PATH = .venv
VENV = .venv
INPUT ?= 100000

$(VENV):
	@poetry install

all: build

build: $(VENV)

train: build
	@clear
	@poetry run python src/train_model.py

estimate: build
	@poetry run python src/estimate_value.py --input $(INPUT)
