#!/bin/bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Set the desired Conda environment name
conda_env_name="groler"

# Check if the Conda environment already exists
if conda info --envs | grep -q "$conda_env_name"; then
    echo "Conda environment '$conda_env_name' already exists."
else
    # Create a new Conda environment with Python 3.8 (you can change the version)
    conda create -y -n "$conda_env_name" python=3.12
fi

# Activate the Conda environment
conda activate "$conda_env_name"  # Use "conda activate" on newer Conda versions

# Install PyTorch (you can specify the version and CUDA support as needed)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.2 matplotlib timer kneed scikit-learn pulp
pip install flash-attn --no-build-isolation
# Old version (2.1)
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# pip install transformers matplotlib timer kneed scikit-learn pulp

# Verify the installation by displaying the PyTorch version
python3 -c "import torch; print('PyTorch version:', torch.__version__)"

# install azure cli
#curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
