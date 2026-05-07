#!/usr/bin/env bash
# litllm — one-line installer.
#
#   curl -sSL https://litllm.dev/install.sh | bash
#
# (or: curl -sSL https://raw.githubusercontent.com/LitLLM/LitLLM/main/skill/install.sh | bash)
#
# Drops the SKILL.md into ~/.claude/skills/litllm/ and installs the `litllm`
# CLI from the LitLLM/LitLLM GitHub repo (or PyPI when available).

set -euo pipefail

REPO="${LITLLM_REPO:-LitLLM/LitLLM}"
BRANCH="${LITLLM_BRANCH:-main}"
SUBDIR="${LITLLM_SUBDIR:-skill}"
SKILL_DIR="${HOME}/.claude/skills/litllm"

echo "litllm installer"
echo "  repo:     ${REPO}@${BRANCH}/${SUBDIR}"
echo "  skill -> ${SKILL_DIR}/SKILL.md"
echo

# 1. Fetch SKILL.md
mkdir -p "${SKILL_DIR}"
curl -fsSL "https://raw.githubusercontent.com/${REPO}/${BRANCH}/${SUBDIR}/SKILL.md" \
  -o "${SKILL_DIR}/SKILL.md"
echo "  [+] SKILL.md installed"

# 2. Install the CLI
PIP_TARGET="git+https://github.com/${REPO}.git@${BRANCH}#subdirectory=${SUBDIR}"
if command -v pipx >/dev/null 2>&1; then
  pipx install --force "${PIP_TARGET}"
  echo "  [+] CLI installed via pipx"
elif command -v pip >/dev/null 2>&1; then
  pip install --user "${PIP_TARGET}"
  echo "  [+] CLI installed via pip --user"
else
  echo "  [!] Need pipx or pip (Python 3.10+). Install one and re-run." >&2
  exit 1
fi

cat <<MSG

litllm is ready.

Set your API key (any OpenAI-compatible endpoint works):
  export LITLLM_API_KEY="sk-..."
  export LITLLM_BASE_URL="https://api.openai.com/v1"   # default
  export LITLLM_MODEL="gpt-4o-mini"                    # default

Then in Claude Code, just say:
  "Find related work for paper.pdf"

Or run the CLI directly:
  litllm related-work paper.pdf
MSG
