all: run-python run-rust run-numpy

run-python:
	uv run python rng_benchmark.py

run-numpy:
	uv run python rng_benchmark_numpy.py

run-rust:
	rustc -O rng_benchmark.rs -o rng_benchmark
	./rng_benchmark
