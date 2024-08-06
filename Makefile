.PHONY: build
build:
	python setup.py sdist

.PHONY: dev
dev:
	pip install -e .

.PHONY: lint
lint:
	python -m black . -l 120
	python -m isort .
	mypy --install-types --non-interactive .

.PHONY: example
example:
	OMP_NUM_THREADS=1 torchrun --nproc-per-node 8 scripts/cifar100_example.py 2>&1 | tee output.txt