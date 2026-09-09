#!/bin/sh
# push / merge 前校验：ruff + mypy + pytest（全量）。任一失败即中止。
set -e

root="$(git rev-parse --show-toplevel)"
venv="$root/.venv"
python="$venv/bin/python"

if [ ! -x "$python" ]; then
    echo "[git 门禁] 缺少 $venv：请先创建 venv 并执行 .venv/bin/pip install -r requirements-dev.txt" >&2
    exit 1
fi

echo "[git 门禁] ruff check…"
"$venv/bin/ruff" check "$root/lang_agent" "$root/tests"

echo "[git 门禁] mypy…"
"$venv/bin/mypy" "$root/lang_agent" "$root/tests" --explicit-package-bases

echo "[git 门禁] pytest…"
"$python" -m pytest -q "$root/tests"

echo "[git 门禁] ✅ 全部通过"
