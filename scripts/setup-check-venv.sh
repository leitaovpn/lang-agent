#!/bin/sh
# 创建旧版 mypy 交叉检查 venv（git 门禁约定为 .venv-check）
# 主 venv 的 mypy 为当前版（requirements-dev.txt），旧版可发现推断差异漏检。
set -e

if ! root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    root="$(cd "$(dirname "$0")/.." && pwd)"
fi
venv="$root/.venv-check"

python3 -m venv "$venv"
"$venv/bin/pip" install -q -U pip
"$venv/bin/pip" install -q -r "$root/requirements-check.txt"
echo "完成：$("$venv/bin/mypy" --version)"
