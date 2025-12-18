# Contributing

Contributions are welcome — bug reports, documentation improvements, new tests, and new features.

## Quick Start

```bash
git clone https://github.com/Alfredo-Sandoval/imgaug2.git
cd imgaug2
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run style/type/security checks:

```bash
ruff check .
ty check .
bandit -c pyproject.toml -r .
```

Build docs locally:

```bash
mkdocs serve
```

## More Details

The canonical contributing guide lives at the repository root:

- `CONTRIBUTING.md`
