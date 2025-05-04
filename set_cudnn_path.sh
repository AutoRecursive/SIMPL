#!/bin/bash
# Define paths
CUDNN_PATH="/home/u24/miniconda3/envs/pred_rl/lib/python3.9/site-packages/torch/lib"
CUDA_PATH="/usr/lib/wsl/lib"
USER_CUDA_PATH="$HOME/cuda/lib64"

# Create user CUDA directory if it doesn't exist
mkdir -p $USER_CUDA_PATH

# Create symbolic links for CUDA libraries
ln -sf $CUDA_PATH/libcuda.so $USER_CUDA_PATH/libcuda.so
ln -sf $CUDA_PATH/libcuda.so.1 $USER_CUDA_PATH/libcuda.so.1

# Create symbolic links for cuDNN libraries
for lib in libcudnn.so.8 libcudnn_adv_infer.so.8 libcudnn_adv_train.so.8 libcudnn_cnn_infer.so.8 libcudnn_cnn_train.so.8 libcudnn_ops_infer.so.8 libcudnn_ops_train.so.8; do
    if [ -f "$CUDNN_PATH/$lib" ]; then
        ln -sf $CUDNN_PATH/$lib $USER_CUDA_PATH/$lib
        echo "Created symbolic link for $lib"
    else
        echo "Warning: $lib not found in $CUDNN_PATH"
    fi
done

# Add paths to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDNN_PATH:$CUDA_PATH:$USER_CUDA_PATH:$LD_LIBRARY_PATH

# Print status
echo "CUDA and cuDNN library paths have been added to LD_LIBRARY_PATH"
echo "Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Verify libraries can be found
echo -e "\nVerifying libraries:"
for lib in libcuda.so libcudnn.so.8 libcudnn_cnn_infer.so.8; do
    if ldconfig -p 2>/dev/null | grep -q $lib || echo $LD_LIBRARY_PATH | tr ':' '\n' | xargs -I{} find {} -name "$lib" 2>/dev/null | grep -q .; then
        echo "✓ $lib is accessible"
    else
        echo "✗ $lib is NOT accessible"
    fi
done
