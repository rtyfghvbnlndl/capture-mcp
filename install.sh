#!/usr/bin/env sh
# capture-mcp installer for Linux / macOS.
#   ./install.sh            install to ~/.capture-mcp/venv
#   ./install.sh --ocr        also install read_screen (screen as text, RapidOCR)
#   ./install.sh --text-only  for models without vision: OCR + CAPTURE_TEXT_ONLY=1 in the client config
#   VENV=/opt/capmcp PYTHON=python3.12 ./install.sh
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$HOME/.capture-mcp/venv}"
PYTHON="${PYTHON:-python3}"
EXTRAS="enum"
ENVJSON="{}"
case "${1:-}" in
  --ocr) EXTRAS="enum,ocr" ;;
  --text-only) EXTRAS="enum,ocr"; ENVJSON='{"CAPTURE_TEXT_ONLY": "1"}' ;;
esac

"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10+ required")'
[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --disable-pip-version-check -q "$HERE[$EXTRAS]"
"$PY" -c "import cv2, mcp, capture_mcp.server; print('import OK, OpenCV', cv2.__version__)"
echo
echo "Video devices:"
"$PY" -m capture_mcp --list || true
[ "$EXTRAS" = "enum,ocr" ] && { echo; echo "OCR self-test:"; "$PY" -m capture_mcp --ocr-test || true; }
cat > "$HERE/mcp-config.json" <<EOF
{
  "mcpServers": {
    "capture-card": {
      "command": "$PY",
      "args": ["-m", "capture_mcp"],
      "env": $ENVJSON
    }
  }
}
EOF
echo
echo "Done. Add this to your MCP client config (also saved to $HERE/mcp-config.json):"
cat "$HERE/mcp-config.json"
# Linux: the user needs access to /dev/video* (usually the "video" group).
