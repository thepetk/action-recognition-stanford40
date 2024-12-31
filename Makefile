.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {sub("\\\\n",sprintf("\n%22c"," "), $$2);printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: run
run:
	uv run main.py

.PHONY: test
test:
	NN_NUM_EPOCHS=1 \
	NN_TRANSFORM_RESIZE=64 \
	SAVE_PLOT=1 \
	NN_TRAIN_BATCH_SIZE=5 \
    NN_TEST_BATCH_SIZE=1 \
    NN_VAL_BATCH_SIZE=1 \
    IMAGE_FILES_PATH="test-resources/img" \
    XML_FILES_PATH="test-resources/xml" \
    SAVE_AS_YAML=0 \
    MODEL="pretrained" \
    TEST_MODE=1 \
    make run

.PHONY: check-type
check-type:
	mypy .

.PHONY: check-format
check-format:
	ruff check --fix

.PHONY: check-all
check-all: check-format check-type test
