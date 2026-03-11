# Tests

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# FP8 attention correctness: Lumen vs TransformerEngine AMD
pytest tests/module/test_fp8_attention.py -v -s
```
