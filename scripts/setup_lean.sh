#!/usr/bin/env bash
set -euo pipefail

echo "Installing Lean 4 toolchain..."

if ! command -v elan &> /dev/null; then
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
    export PATH="$HOME/.elan/bin:$PATH"
fi

elan default leanprover/lean4:stable
lean --version
lake --version

echo "Lean toolchain installation complete."
