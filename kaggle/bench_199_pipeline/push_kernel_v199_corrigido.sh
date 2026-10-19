#!/usr/bin/env bash
# push_kernel_v199_corrigido.sh — Envia o Winnex Pipeline Honest Bench v199 CORRIGIDO ao Kaggle.
#
# Diferença vs o v199 original (winnex-pipeline-honest-bench-v199):
#   corrige os gargalos da análise — GT externo ligado ao certificado, logs
#   visíveis, word2vec real+simulado separados, latência com/sem scan,
#   effective_basis fiel, baseline numpy.
#
# Requisito: credencial Kaggle com permissão de KERNELS.
#   - Se der 401: gere novo token em https://www.kaggle.com/settings (com permissão
#     de kernels) e substitua ~/.kaggle/kaggle.json.
#
# Executar de dentro deste diretório:
#   cd winnex-madhava/kaggle/bench_199_pipeline && ./push_kernel_v199_corrigido.sh

set -euo pipefail
cd "$(dirname "$0")"

CODE=winnex-pipeline-honest-v199-corrigido.py
META=kernel-metadata-v199-corrigido.json
SLUG="kleniopadilha/winnex-pipeline-honest-bench-v199-corrigido"

echo "=== Winnex Pipeline Honest Bench v199 CORRIGIDO (madhava 1.9.9 + normalize 1.2.1) ==="
echo "Arquivo: $CODE ($(wc -l < "$CODE") linhas)"
echo "Slug:    $SLUG"
echo

if ! kaggle kernels list --mine >/dev/null 2>&1; then
    echo "ERRO: a credencial atual não tem acesso à API de kernels (401)."
    echo "  - Gere um novo token em https://www.kaggle.com/settings"
    echo "  - Substitua ~/.kaggle/kaggle.json"
    exit 1
fi

echo "Credencial de kernels OK. Enviando (metadata: $META)..."
# O `kaggle kernels push` usa o arquivo kernel-metadata.json (nome fixo). Como o
# original do v199 já ocupa kernel-metadata.json, usamos um diretório temporário
# com o metadata renomeado para o nome esperado pelo CLI.
TMPDIR=$(mktemp -d)
cp "$CODE" "$TMPDIR/$CODE"
cp "$META" "$TMPDIR/kernel-metadata.json"
kaggle kernels push -p "$TMPDIR"
rm -rf "$TMPDIR"
echo
echo "Acompanhe: kaggle kernels status $SLUG"
echo "Log:       kaggle kernels output $SLUG"
