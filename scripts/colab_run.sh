#!/usr/bin/env bash
# ============================================================
# PureReason — Google Colab T4 (16GB) single-script launcher
# Run in a Colab cell as:
#   !bash colab_run.sh
# ============================================================
set -euo pipefail

# --- Python deps ---
pip install -q torch transformers trl peft bitsandbytes datasets accelerate sympy wandb

# --- Lean + Mathlib ---
if ! command -v lean &>/dev/null; then
    curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
fi
export PATH="$HOME/.elan/bin:$PATH"
elan default leanprover/lean4:stable

if [ ! -d /tmp/lean_verify/.lake ]; then
    cd /tmp && rm -rf lean_verify && mkdir lean_verify && cd lean_verify
    cat > lakefile.lean << 'EOF'
import Lake
open Lake DSL
require mathlib from git "https://github.com/leanprover-community/mathlib4"
package «lean_verify» where moreLeanArgs := #["-DwarningAsError=false"]
@[default_target] lean_lib «LeanVerify» where
EOF
    lake update && lake exe cache get
fi

# --- Training ---
cd /content/purereason 2>/dev/null || cd "$(dirname "$0")/.."
python src/train.py --preset colab
