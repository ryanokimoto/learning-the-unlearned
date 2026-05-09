#!/bin/bash
# setup_verification.sh
# Run this script to verify your environment is ready for baseline reproduction

echo "================================"
echo "Environment Verification Script"
echo "================================"
echo ""

# Check Python version
echo "1. Checking Python version..."
python_version=$(python --version 2>&1)
echo "   $python_version"
if [[ $python_version == *"3.10"* ]]; then
    echo "   ✓ Python 3.10 found"
else
    echo "   ✗ Warning: Python 3.10 recommended"
fi
echo ""

# Check CUDA availability
echo "2. Checking CUDA availability..."
cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>&1)
if [[ $cuda_available == "True" ]]; then
    echo "   ✓ CUDA is available"
    gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
    echo "   ✓ Number of GPUs: $gpu_count"
    gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "   ✓ GPU: $gpu_name"
else
    echo "   ✗ CUDA not available - GPU required for training"
fi
echo ""

# Check required packages
echo "3. Checking required packages..."
packages=("torch" "transformers" "datasets" "accelerate" "peft" "hydra-core" "deepspeed")
all_installed=true

for package in "${packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        version=$(python -c "import $package; print($package.__version__ if hasattr($package, '__version__') else 'installed')")
        echo "   ✓ $package ($version)"
    else
        echo "   ✗ $package not found"
        all_installed=false
    fi
done
echo ""

# Check disk space
echo "4. Checking disk space..."
disk_space=$(df -h . | awk 'NR==2 {print $4}')
echo "   Available space: $disk_space"
echo "   (Recommended: >500GB for model checkpoints)"
echo ""

# Check HuggingFace cache
echo "5. Checking HuggingFace cache..."
if [ -z "$HF_HOME" ]; then
    echo "   HF_HOME not set (using default ~/.cache/huggingface)"
else
    echo "   HF_HOME: $HF_HOME"
fi
echo ""

# Check repository files
echo "6. Checking repository files..."
required_files=(
    "jog_llm_memory/synthetic/finetune.py"
    "jog_llm_memory/synthetic/forget.py"
    "jog_llm_memory/synthetic/relearn.py"
    "jog_llm_memory/synthetic/eval.py"
    "jog_llm_memory/synthetic/common_names_ft.txt"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file not found"
    fi
done
echo ""

# Summary
echo "================================"
if [[ $cuda_available == "True" ]] && [[ $all_installed == true ]]; then
    echo "✓ Environment ready for baseline reproduction!"
    echo ""
    echo "Next steps:"
    echo "1. cd jog_llm_memory/synthetic"
    echo "2. Run: bash ../scripts/run_baseline.sh"
else
    echo "✗ Please fix the issues above before proceeding"
    echo ""
    echo "Installation command:"
    echo "conda create -n synthetic python=3.10"
    echo "conda activate synthetic"
    echo "conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia"
    echo "pip install -r requirements.txt"
fi
echo "================================"