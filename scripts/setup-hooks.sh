#!/bin/bash
set -e

echo "Configuring CarlaBEV quality hooks..."

if ! command -v pre-commit &> /dev/null; then
  echo "Installing pre-commit tool..."
  uv tool install pre-commit
fi

echo "Installing pre-commit and pre-push hooks..."
uv tool run pre-commit install
uv tool run pre-commit install --hook-type pre-push

echo "Quality hooks active."
