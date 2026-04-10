# Contributing to OnlineCML

Thank you for your interest in contributing! This guide covers everything
you need to get started.

## Ways to Contribute

- **Bug reports** — open an issue with a minimal reproducible example
- **New estimators** — see [Adding a New Estimator](adding_estimator.md)
- **Documentation** — fix typos, improve examples, add theory
- **Tests** — add edge cases or regression tests
- **Benchmarks** — add comparisons with new methods or datasets

## Development Setup

```bash
# Clone the repository
git clone https://github.com/athammad/onlinecml
cd onlinecml

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the test suite
pytest tests/ -v

# Run linting
ruff check onlinecml/

# Run type checking
mypy onlinecml/
```

## Code Standards

### Style

All code is formatted and linted with [ruff](https://docs.astral.sh/ruff/).
Run `ruff check --fix onlinecml/` before committing.

Key rules (see `pyproject.toml`):

- Line length: 100 characters
- Imports: isort-style (`I`)
- Flake8 errors (`E`, `F`)
- Bugbear (`B`)
- Naming (`N`) — uppercase `X`, `W`, `Y` allowed (causal convention)

### Type Hints

All public methods must have type hints. Run `mypy onlinecml/` to check.

### Docstrings

Every public class and method requires a NumPy-style docstring with
`Parameters`, `Returns`, and `Examples` sections. See any existing class
for the standard format.

## Testing Requirements

Every pull request must include:

1. **Unit tests** — one test file per class, in `tests/unit/`
2. **At minimum:** `test_n_seen_*`, `test_predict_one_*`, `test_reset_*`, `test_clone_*`
3. **Coverage** — the test suite must maintain ≥ 90% coverage

Run tests with:

```bash
pytest tests/unit/ tests/integration/ tests/regression/ -v
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Implement your change with tests and docstrings
3. Run the full test suite and ensure it passes
4. Open a pull request with a clear description of the change
5. Link any related issues

## Issue Labels

| Label | Meaning |
|---|---|
| `bug` | Something is broken |
| `feature` | New estimator or method |
| `research` | Theoretical question |
| `docs` | Documentation improvement |
| `good first issue` | Beginner-friendly task |

## Code of Conduct

This project follows the
[Contributor Covenant](https://www.contributor-covenant.org/) v2.1.
Be respectful, constructive, and welcoming.
