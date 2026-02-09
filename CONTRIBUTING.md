# Contributing to OneValet

Thanks for your interest in contributing to OneValet!

## Development Setup

```bash
# Clone the repo
git clone https://github.com/xiaoyu-work/onevalet.git
cd onevalet

# Install in development mode
pip install -e ".[all]"

# Install dev dependencies
pip install -r requirements-dev.txt
```

## Running Tests

```bash
pytest
```

## Code Style

We use [Black](https://github.com/psf/black) for formatting and [Ruff](https://github.com/astral-sh/ruff) for linting.

```bash
# Format code
black onevalet tests

# Lint
ruff check onevalet tests

# Type check
mypy onevalet
```

## Pull Request Process

1. Fork the repo and create your branch from `main`
2. Add tests for any new functionality
3. Ensure all tests pass
4. Update documentation if needed
5. Submit PR with a clear description

## Commit Messages

Use clear, descriptive commit messages:

```
Add restaurant booking example
Fix validation error in InputField
Update getting-started documentation
```

## Reporting Issues

- Use the issue templates
- Include steps to reproduce
- Include Python version and OS

## Questions?

Open an issue with the `question` label.
