# Contributing

We welcome contributions to imgaug2!

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a virtual environment
4. Install development dependencies

```bash
git clone https://github.com/YOUR_USERNAME/imgaug2.git
cd imgaug2
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Code Style

We use `ruff` for linting:

```bash
ruff check .
ruff format .
```

## Pull Requests

1. Create a feature branch
2. Make your changes
3. Add tests if applicable
4. Run the test suite
5. Submit a pull request

## Reporting Issues

Use the [GitHub issue tracker](https://github.com/Alfredo-Sandoval/imgaug2/issues) for:

- Bug reports
- Feature requests
- Questions
