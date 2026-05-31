#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks

chmod +x .githooks/pre-commit .githooks/pre-push

echo "Configured git hooks at .githooks"
