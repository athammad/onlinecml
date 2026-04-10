# Installation

## Requirements

- Python 3.10 or higher
- Dependencies installed automatically: `river>=0.23`, `numpy>=2.0`, `scipy>=1.10`, `matplotlib>=3.7`

## Install from PyPI

```bash
pip install onlinecml
```

## Install for development

```bash
git clone https://github.com/athammad/onlinecml
cd onlinecml
pip install -e ".[dev]"
```

## Install with docs dependencies

```bash
pip install -e ".[docs]"
mkdocs serve  # live preview at http://127.0.0.1:8000
```

## Verify

```python
import onlinecml
print(onlinecml.__version__)  # 1.0.0
```
