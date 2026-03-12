#!/bin/bash
set -euo pipefail

REPO_URL="https://github.com/benson-guo/zorse.git"
REPO_DIR="zorse"
MINICONDA_DIR="$HOME/miniconda3"
MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
CONDA_ENV_NAME="groler"
PYTHON_VERSION="3.12"

# ─── Clone zorse repo ────────────────────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    echo "[✓] Repo '$REPO_DIR' already cloned — pulling latest changes."
    git -C "$REPO_DIR" pull --ff-only || echo "[!] Pull failed (non-fatal); continuing with existing checkout."
else
    echo "[…] Cloning $REPO_URL"
    git clone "$REPO_URL" "$REPO_DIR"
    echo "[✓] Clone complete."
fi

# ─── Ensure Miniconda is installed ────────────────────────────────────
install_miniconda() {
    echo "[…] Installing Miniconda to $MINICONDA_DIR"
    local arch
    arch="$(uname -m)"
    local os
    os="$(uname -s)"

    local url_base="https://repo.anaconda.com/miniconda"
    local installer_name=""

    case "$os" in
        Linux)
            case "$arch" in
                x86_64)  installer_name="Miniconda3-latest-Linux-x86_64.sh" ;;
                aarch64) installer_name="Miniconda3-latest-Linux-aarch64.sh" ;;
                *)       echo "[✗] Unsupported Linux arch: $arch"; exit 1 ;;
            esac
            ;;
        Darwin)
            case "$arch" in
                x86_64)  installer_name="Miniconda3-latest-MacOSX-x86_64.sh" ;;
                arm64)   installer_name="Miniconda3-latest-MacOSX-arm64.sh" ;;
                *)       echo "[✗] Unsupported macOS arch: $arch"; exit 1 ;;
            esac
            ;;
        *)
            echo "[✗] Unsupported OS: $os"; exit 1
            ;;
    esac

    local url="${url_base}/${installer_name}"

    if command -v curl &>/dev/null; then
        curl -fsSL -o "$MINICONDA_INSTALLER" "$url"
    elif command -v wget &>/dev/null; then
        wget -qO "$MINICONDA_INSTALLER" "$url"
    else
        echo "[✗] Neither curl nor wget found — cannot download Miniconda."
        exit 1
    fi

    bash "$MINICONDA_INSTALLER" -b -p "$MINICONDA_DIR"
    rm -f "$MINICONDA_INSTALLER"
    echo "[✓] Miniconda installed."
}

if [ -x "$MINICONDA_DIR/bin/conda" ]; then
    echo "[✓] Miniconda already installed at $MINICONDA_DIR"
elif command -v conda &>/dev/null; then
    MINICONDA_DIR="$(conda info --base)"
    echo "[✓] Found existing conda at $MINICONDA_DIR"
else
    install_miniconda
fi

# ─── Initialise conda for this shell session ──────────────────────────
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# ─── Create / activate conda environment ──────────────────────────────
if conda info --envs | grep -qw "$CONDA_ENV_NAME"; then
    echo "[✓] Conda environment '$CONDA_ENV_NAME' already exists."
else
    echo "[…] Creating conda environment '$CONDA_ENV_NAME' (Python $PYTHON_VERSION)"
    conda create -y -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION"
fi

conda activate "$CONDA_ENV_NAME"
echo "[✓] Activated environment '$CONDA_ENV_NAME' ($(python --version))"

# ─── Install Python packages ─────────────────────────────────────────
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.2 matplotlib timer kneed scikit-learn pulp pymetis
# ─── Flash Attention (GPU-dependent) ──────────────────────────────────
# flash-attn v2 requires Ampere or newer (SM ≥ 80): A100, A10, RTX 30xx/40xx, H100 …
# Older GPUs like T4 (SM 75), V100 (SM 70), P100 (SM 60) are NOT compatible with v2.
install_flash_attn() {
    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "[!] No CUDA device detected — skipping flash-attn."
        return
    fi

    local gpu_name sm_major
    gpu_name="$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")"
    sm_major="$(python3 -c "import torch; print(torch.cuda.get_device_capability(0)[0])")"
    echo "[…] Detected GPU: $gpu_name  (SM ${sm_major}.x)"

    if [ "$sm_major" -ge 8 ]; then
        echo "[…] SM ≥ 8.0 — installing latest flash-attn (v2)."
        pip install flash-attn --no-build-isolation \
            || echo "[!] flash-attn v2 build failed — skipping."
    elif [ "$sm_major" -eq 7 ]; then
        echo "[!] SM 7.x GPU ($gpu_name) is NOT supported by flash-attn v2."
        echo "    T4 (SM 7.5) and V100 (SM 7.0) users: flash-attn will be skipped."
        echo "    The model will fall back to standard scaled-dot-product attention."
    else
        echo "[!] SM ${sm_major}.x is too old for any flash-attn version — skipping."
    fi
}
install_flash_attn

# ─── Verify ───────────────────────────────────────────────────────────
python3 -c "import torch; print('[✓] PyTorch version:', torch.__version__)"
echo "[✓] Setup complete."
