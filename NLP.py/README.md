# NLP Practical 1: Text Preprocessing

This practical demonstrates the following preprocessing steps:

1. Tokenization
2. Filtration
3. Script Validation
4. Stop-word Removal
5. Stemming

## Run

```bash
python3 text_preprocessing_practical.py
```

## What the script does

- Tokenizes input text into words.
- Filters tokens using:
  - lowercase normalization
  - minimum length rule
  - removal of numeric-only tokens
  - alphabetic presence checks
- Validates that tokens belong to allowed scripts (default: `LATIN`).
- Removes common stop words.
- Applies lightweight stemming rules.

## Customize

You can edit the `PreprocessingConfig` in `main()` to change:

- `allowed_scripts`
- `min_token_length`
- `remove_numeric_tokens`
- `stopwords`
