VENV_PATH = ./.venv

$(VENV_PATH):
	@poetry install

all: run

build: $(VENV_PATH)

run: build
	@poetry run python src/main.py
