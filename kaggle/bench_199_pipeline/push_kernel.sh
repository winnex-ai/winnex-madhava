#!/usr/bin/env bash
# push_kernel.sh — Submete o Winnex Pipeline Honest Bench v199 ao Kaggle.
#
# Requisito: credencial Kaggle com permissão de KERNELS.
# Se o token atual der 401 Unauthorized para kernels:
#   1. Abra https://www.kaggle.com/settings
#   2. Gere um novo API Token (Create New API Token) — o token default é
#      datasets-only; kernels push precisa do token com permissão de kernels.
#   3. Substitua ~/.kaggle/kaggle.json pelo novo token
#   4. Teste com: kaggle kernels list --mine
#   5. Rode este script.
#
# Executar de dentro deste diretório:
#   cd winnex-madhava/kaggle/bench_199_pipeline && ./push_kernel.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Winnex Pipeline Honest Bench v199 (madhava 1.9.9 + normalize 1.2.1) ==="
echo "Arquivo: winnex-pipeline-1-9-9-honest.py ($(wc -l < winnex-pipeline-1-9-9-honest.py) linhas)"
echo "Slug:    kleniopadilha/winnex-pipeline-honest-bench-v199"
echo

if ! kaggle kernels list --mine >/dev/null 2>&1; then
    echo "ERRO: a credencial atual não tem acesso à API de kernels (401)."
    echo "  - Gere um novo token em https://www.kaggle.com/settings"
    echo "  - Substitua ~/.kaggle/kaggle.json"
    echo "  - Teste: kaggle kernels list --mine"
    exit 1
fi

echo "Credencial de kernels OK. Enviando..."
kaggle kernels push -p ./
echo
echo "Acompanhe: kaggle kernels status kleniopadilha/winnex-pipeline-honest-bench-v199"
echo "Log:       kaggle kernels output kleniopadilha/winnex-pipeline-honest-bench-v199"
