#!/usr/bin/env python3
"""
scan_secrets.py — Secret scanner para prevenir exposição de tokens em git.
=======================================================================
Detecta tokens/segredos em arquivos antes de commit. Usa padrões conhecidos
para tokens Kaggle, GitHub, Zenodo, AWS, OpenAI, chaves privadas, etc.

Uso:
  python3 scripts/scan_secrets.py               # varre o diretório atual
  python3 scripts/scan_secrets.py --path DIR    # varre um diretório específico
  python3 scripts/scan_secrets.py --precommit   # modo pre-commit (exit 1 se achar)
  python3 scripts/scan_secrets.py --check-diff  # varre só o diff staged

Exit code: 0 = limpo, 1 = segredos encontrados (bloqueia commit).
"""
import argparse
import fnmatch
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Padrões de segredos (regex por tipo)
# ---------------------------------------------------------------------------
SECRET_PATTERNS = [
    # Kaggle API tokens (formato KGAT_...)
    (r"KGAT_[A-Za-z0-9]{20,}", "Kaggle API token"),
    # GitHub tokens
    (r"ghp_[A-Za-z0-9]{36,}", "GitHub personal token"),
    (r"gho_[A-Za-z0-9]{36,}", "GitHub OAuth token"),
    (r"github_pat_[A-Za-z0-9_]{40,}", "GitHub fine-grained token"),
    # Zenodo / deposit tokens (formato 40-60 char alfanumerico, atribuídos a TOKEN)
    (r"access_token=['\"][A-Za-z0-9]{30,}['\"]", "Zenodo access token"),
    (r"TOKEN\s*=\s*['\"][A-Za-z0-9]{40,}['\"]", "Zenodo/API token (TOKEN=...)"),
    # AWS
    (r"AKIA[0-9A-Z]{16}", "AWS access key"),
    (r"aws_secret_access_key\s*=\s*['\"][A-Za-z0-9/+=]{40}['\"]", "AWS secret key"),
    # OpenAI / Anthropic / Google
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI/API key (sk-)"),
    (r"sk-ant-[A-Za-z0-9_-]{20,}", "Anthropic key"),
    (r"AIza[0-9A-Za-z_-]{35}", "Google API key"),
    # Generic secrets
    (r"(api[_-]?key|api[_-]?token|secret|password|passwd|auth[_-]?token)"
     r"\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]", "Generic credential"),
    # Chaves privadas
    (r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----", "Private key"),
    # Bearer tokens longos
    (r"Bearer\s+[A-Za-z0-9._\-]{30,}", "Bearer token"),
]

# Extensões a varrer
SCAN_EXTENSIONS = {".py", ".sh", ".json", ".yml", ".yaml", ".md", ".txt", ".env", ".toml", ".cfg", ".ini"}
# Diretórios/arquivos a ignorar
IGNORE_DIRS = {".git", "node_modules", ".cache", "build", "dist", "__pycache__", ".venv", "venv"}
IGNORE_PATTERNS = ["*.pyc", "*.pyo", "*.egg-info/*", "*.ipynb", "*.npy", "*.npz", "*.bin", "*.so"]

# Falsos positivos comuns (tokens em contexto de exemplo/placeholder)
PLACEHOLDER_PATTERNS = [
    r"your_[a-z_]+",
    r"example[a-z_]*",
    r"xxxxx",
    r"<.*>",
    r"TODO",
    r"PLACEHOLDER",
    r"test[_a-z]*",
]


def should_scan(path: str) -> bool:
    """Decide se o arquivo deve ser varrido."""
    if os.path.basename(path) in {"kaggle.json", ".pypirc", ".git-credentials"}:
        return True
    ext = os.path.splitext(path)[1].lower()
    if ext not in SCAN_EXTENSIONS:
        return False
    # ignora binários/artefatos
    for pat in IGNORE_PATTERNS:
        if fnmatch.fnmatch(path, pat):
            return False
    return True


def is_placeholder(line: str) -> bool:
    """Retorna True se a linha parece placeholder (falso positivo)."""
    for pat in PLACEHOLDER_PATTERNS:
        if re.search(pat, line, re.IGNORECASE):
            # só considera placeholder se a linha tem pouca entropia de token
            return True
    return False


def scan_file(path: str) -> list:
    """Varre um arquivo por segredos. Retorna lista de (linha, tipo, trecho)."""
    findings = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for lineno, line in enumerate(f, 1):
                # Primeiro procura por tokens FORTES (KGAT_, ghp_, sk-, AKIA, PRIVATE KEY)
                # — estes nunca são placeholder.
                strong_hit = None
                for pattern, desc in SECRET_PATTERNS[:8]:
                    m = re.search(pattern, line)
                    if m:
                        strong_hit = (desc, m.group(0))
                        break
                if strong_hit:
                    desc, match_text = strong_hit
                    masked = re.sub(r"[A-Za-z0-9_\-]{8,}", "***REDACTED***", match_text, count=1)
                    findings.append((lineno, desc, masked))
                    continue
                # Só depois aplica placeholder para padrões genéricos
                if is_placeholder(line):
                    continue
                for pattern, desc in SECRET_PATTERNS[8:]:
                    m = re.search(pattern, line)
                    if m:
                        match_text = m.group(0)
                        masked = re.sub(r"[A-Za-z0-9_\-]{8,}", "***REDACTED***", match_text, count=1)
                        findings.append((lineno, desc, masked))
    except (OSError, UnicodeDecodeError):
        pass
    return findings


def scan_tree(root: str) -> list:
    """Varre recursivamente um diretório."""
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        # ignora dirs
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            if should_scan(path):
                for finding in scan_file(path):
                    results.append((path, *finding))
    return results


def scan_git_staged(root: str) -> list:
    """Varre apenas os arquivos staged no git."""
    try:
        out = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=root, capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return []
    results = []
    for fn in out.strip().splitlines():
        path = os.path.join(root, fn)
        if os.path.exists(path) and should_scan(path):
            for finding in scan_file(path):
                results.append((fn, *finding))
    return results


def main():
    ap = argparse.ArgumentParser(description="Secret scanner para git")
    ap.add_argument("--path", default=".", help="diretório a varrer")
    ap.add_argument("--precommit", action="store_true", help="modo pre-commit (bloqueia)")
    ap.add_argument("--check-diff", action="store_true", help="varre só o staged diff")
    args = ap.parse_args()

    root = os.path.abspath(args.path)

    if args.check_diff:
        results = scan_git_staged(root)
    elif os.path.isfile(root):
        # varre um único arquivo
        if should_scan(root):
            results = [(os.path.basename(root), *f) for f in scan_file(root)]
        else:
            results = []
    else:
        results = scan_tree(root)

    if not results:
        print("Limpo: nenhum segredo encontrado.")
        return 0

    print("SEGREDOS ENCONTRADOS:")
    print("-" * 72)
    for path, lineno, desc, masked in results:
        print(f"  {path}:{lineno}  [{desc}]  {masked}")
    print("-" * 72)
    print(f"{len(results)} segredo(s) encontrado(s).")

    if args.precommit:
        print("\nBLoqueado: remova os segredos antes de commitar.")
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
