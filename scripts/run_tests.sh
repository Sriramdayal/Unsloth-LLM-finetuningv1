#!/bin/bash
# Run fast tests (excludes slow/gpu tests) with coverage reporting

set -e

echo "===================================================="
echo "Running Fast Test Suite (excluding slow & GPU tests)"
echo "===================================================="

uv run pytest -m "not slow and not gpu" --cov=src --cov-report=term-missing --cov-report=html tests/

echo "===================================================="
echo "Fast test suite run completed successfully!"
echo "HTML coverage report generated in htmlcov/index.html"
echo "===================================================="
