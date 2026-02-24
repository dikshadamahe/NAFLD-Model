# Git Workflow Reference - NAFLD-Model

## Branching Strategy

```bash
# Create branches
git branch dev
git branch experiment

# Switch to dev for active development
git checkout dev

# Switch back to main
git checkout main

# Merge dev into main (after testing)
git checkout main
git merge dev --no-ff -m "merge: integrate dev into main"

# Delete experiment branch after use
git branch -d experiment
```

## Tagging for Research Milestones

```bash
# After preprocessing complete
git tag -a v1.0-preprocessing -m "Preprocessing pipeline complete"

# After model comparison complete
git tag -a v2.0-model-comparison -m "24 ML models trained and compared"

# After best model selection
git tag -a v3.0-final-model -m "Best model selected and saved"

# Push all tags
git push origin --tags
```

## Commit Message Format

```
<type>: <short summary>

Types:
  feat     - new feature / module
  fix      - bug fix
  docs     - documentation
  refactor - code restructuring
  test     - adding tests
  chore    - maintenance tasks
```

## Safe Push Workflow

```bash
chmod +x push_changes.sh
./push_changes.sh
```
