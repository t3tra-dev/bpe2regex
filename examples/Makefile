PROJECT_ROOT := ..
PYTHON ?= python3
NODE ?= node
BPE2REGEX ?= .venv/bin/bpe2regex
ARGS ?=
LOG_LEVEL ?= INFO

.PHONY: all artifacts verify run

all: verify

artifacts:
	cd $(PROJECT_ROOT) && $(BPE2REGEX) build r50k --force --log-level $(LOG_LEVEL)
	cd $(PROJECT_ROOT) && $(BPE2REGEX) build p50k --force --log-level $(LOG_LEVEL)
	cd $(PROJECT_ROOT) && $(BPE2REGEX) build cl100k --force --log-level $(LOG_LEVEL)
	cd $(PROJECT_ROOT) && $(BPE2REGEX) build o200k --force --log-level $(LOG_LEVEL)

verify: artifacts
	$(PYTHON) python.py --artifact ../.artifacts/r50k/python.bin --verify
	$(NODE) javascript.mjs --artifact ../.artifacts/r50k/ecmascript.bin --verify
	$(PYTHON) python.py --artifact ../.artifacts/p50k/python.bin --verify
	$(NODE) javascript.mjs --artifact ../.artifacts/p50k/ecmascript.bin --verify
	$(PYTHON) python.py --artifact ../.artifacts/cl100k/python.bin --verify
	$(NODE) javascript.mjs --artifact ../.artifacts/cl100k/ecmascript.bin --verify
	$(PYTHON) python.py --artifact ../.artifacts/o200k/python.bin --verify
	$(NODE) javascript.mjs --artifact ../.artifacts/o200k/ecmascript.bin --verify
	cd $(PROJECT_ROOT) && .venv/bin/python tests/cross_language.py

run: LOG_LEVEL := WARNING
run: artifacts
	$(PYTHON) python.py --artifact ../.artifacts/r50k/python.bin --quiet $(ARGS)
	$(NODE) javascript.mjs --artifact ../.artifacts/r50k/ecmascript.bin $(ARGS)
	$(PYTHON) python.py --artifact ../.artifacts/p50k/python.bin --quiet $(ARGS)
	$(NODE) javascript.mjs --artifact ../.artifacts/p50k/ecmascript.bin $(ARGS)
	$(PYTHON) python.py --artifact ../.artifacts/cl100k/python.bin --quiet $(ARGS)
	$(NODE) javascript.mjs --artifact ../.artifacts/cl100k/ecmascript.bin $(ARGS)
	$(PYTHON) python.py --artifact ../.artifacts/o200k/python.bin --quiet $(ARGS)
	$(NODE) javascript.mjs --artifact ../.artifacts/o200k/ecmascript.bin $(ARGS)
