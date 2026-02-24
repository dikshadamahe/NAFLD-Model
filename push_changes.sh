#!/bin/bash
# ============================================================
# push_changes.sh
# Safe Git Push Script for NAFLD-Model Research Project
# ============================================================

set -e

echo "============================================"
echo "  NAFLD-Model - Safe Git Push Script"
echo "============================================"
echo ""

# Step 1: Check git status
echo "[1/5] Checking git status..."
STATUS=$(git status --porcelain)

if [ -z "$STATUS" ]; then
    echo "✅ No changes to commit. Working tree is clean."
    exit 0
fi

echo "Changed files:"
git status --short
echo ""

# Step 2: Stage all changes
echo "[2/5] Staging all changes..."
git add -A
echo "✅ All changes staged."
echo ""

# Step 3: Ask user for commit message
read -p "[3/5] Enter commit message: " COMMIT_MSG

if [ -z "$COMMIT_MSG" ]; then
    echo "❌ Error: Commit message cannot be empty. Aborting."
    git reset HEAD -- . > /dev/null 2>&1
    exit 1
fi

# Step 4: Commit
echo "[4/5] Committing changes..."
git commit -m "$COMMIT_MSG"
echo "✅ Changes committed."
echo ""

# Step 5: Push to origin main
echo "[5/5] Pushing to origin main..."
git push origin main
echo ""
echo "============================================"
echo "  ✅ Push completed successfully!"
echo "============================================"
