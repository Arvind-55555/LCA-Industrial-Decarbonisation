# Contributing Guide

## Development Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Code Style

- Follow PEP 8
- Use `black` for formatting
- Use `flake8` for linting
- Type hints are encouraged

## Testing

Run tests:
```bash
pytest tests/
```

With coverage:
```bash
pytest tests/ --cov=lca_optimizer --cov-report=html
```

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Add tests
4. Ensure all tests pass
5. Submit a pull request

## Adding New Sectors

To add a new sector:

1. Create a new file in `lca_optimizer/sectors/`
2. Implement sector-specific optimizer class
3. Add API endpoint in `lca_optimizer/api/endpoints.py`
4. Add tests in `tests/`

