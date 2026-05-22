#!/bin/bash
# Run all tests in the suite with coverage reporting

set -e

echo "===================================================="
echo "Running Full Test Suite"
echo "===================================================="

uv run pytest --cov=src --cov-report=term-missing --cov-report=html tests/

echo "===================================================="
echo "Full test suite run completed successfully!"
echo "HTML coverage report generated in htmlcov/index.html"
echo "===================================================="
