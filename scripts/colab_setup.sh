#!/usr/bin/env bash
# Google Colab T4 setup for PureReason
# Run this cell first in Colab:
#   !curl -O https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh
#   !bash elan-init.sh -y
# Then run: !bash scripts/colab_setup.sh

set -euo pipefail

echo "=== Installing Python dependencies ==="
pip install -q torch transformers trl peft bitsandbytes datasets accelerate sympy wandb

echo "=== Setting up Lean + Mathlib ==="
export PATH="$HOME/.elan/bin:$PATH"
elan default leanprover/lean4:stable
lean --version

cd /tmp
rm -rf lean_verify
mkdir lean_verify
cd lean_verify

cat > lakefile.lean << 'LAKEEOF'
import Lake
open Lake DSL
require mathlib from git "https://github.com/leanprover-community/mathlib4"
package «lean_verify» where moreLeanArgs := #["-DwarningAsError=false"]
@[default_target] lean_lib «LeanVerify» where
LAKEEOF

lake update
lake exe cache get

echo "=== Setup complete ==="
echo "Run: python src/train.py --preset colab"
