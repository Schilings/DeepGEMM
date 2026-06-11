#!/bin/bash
# ============================================================
# Nsight Compute (ncu) 安装脚本
# 适用于 Ubuntu 容器（AWS Koala 训练平台等）
# AI 可直接执行: bash dev/startup/install_ncu.sh
# ============================================================

set -e

echo "========================================="
echo "安装 Nsight Compute (ncu)"
echo "========================================="

# 检查是否已安装
if command -v ncu &> /dev/null; then
    echo "✅ ncu 已安装: $(ncu --version 2>/dev/null || echo 'version unknown')"
    echo "   路径: $(which ncu)"
    exit 0
fi

# 1. 确保 apt 更新
apt update -y

# 2. 添加 NVIDIA devtools repo（如果还没加）
if [ ! -f /etc/apt/sources.list.d/nvidia-devtools.list ]; then
    echo "添加 NVIDIA devtools 仓库..."
    echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release 2>/dev/null && echo "$DISTRIB_RELEASE" | tr -d . || echo "2204")/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
else
    echo "NVIDIA devtools 仓库已存在，跳过"
fi

# 3. 安装 nsight-compute
echo "安装 nsight-compute..."
apt install -y nsight-compute-cli 2>/dev/null || \
apt install -y nsight-compute 2>/dev/null || {
    echo "⚠️  apt 安装失败，尝试从 CUDA toolkit 获取..."
    # 回退方案：检查 CUDA 自带的 ncu
    if [ -f /usr/local/cuda/bin/ncu ]; then
        echo "✅ 在 CUDA toolkit 中找到 ncu: /usr/local/cuda/bin/ncu"
        ln -sf /usr/local/cuda/bin/ncu /usr/local/bin/ncu 2>/dev/null || true
        exit 0
    fi
    # 回退方案2：下载 standalone 版本
    echo "下载 Nsight Compute standalone..."
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' || echo "13.0")
    NCUR_VERSION="2025.1.1"
    NCUR_PKG="nsight-compute-linux-x86_64-${NCUR_VERSION}"
    cd /tmp
    wget -q "https://developer.download.nvidia.com/compute/nsight-compute/${NCUR_VERSION}/local_installers/${NCUR_PKG}.run" -O ncu_installer.run || {
        echo "❌ 下载失败，请手动安装: https://developer.nvidia.com/nsight-compute"
        exit 1
    }
    chmod +x ncu_installer.run
    ./ncu_installer.run -q -d /opt/nvidia/nsight-compute
    ln -sf /opt/nvidia/nsight-compute/ncu /usr/local/bin/ncu
    rm -f ncu_installer.run
    cd -
}

# 4. 更新 apt 缓存并再次尝试
apt update -y

# 5. 验证安装
if command -v ncu &> /dev/null; then
    echo "✅ ncu 安装成功: $(ncu --version 2>/dev/null || echo 'installed')"
    echo "   路径: $(which ncu)"
else
    # 可能安装在非标准路径，找一下
    NCU_PATH=$(find /opt/nvidia /usr/local/cuda -name "ncu" -type f 2>/dev/null | head -1)
    if [ -n "$NCU_PATH" ]; then
        ln -sf "$NCU_PATH" /usr/local/bin/ncu
        echo "✅ 找到 ncu: $NCU_PATH → 已链接到 /usr/local/bin/ncu"
    else
        echo "❌ 安装失败，请手动安装"
        exit 1
    fi
fi

# 6. 添加到 PATH（如果还没加）
if ! echo "$PATH" | grep -q "nsight-compute"; then
    NCU_DIR=$(dirname "$(which ncu 2>/dev/null)" 2>/dev/null || echo "")
    if [ -n "$NCU_DIR" ] && [ "$NCU_DIR" != "/usr/bin" ] && [ "$NCU_DIR" != "/usr/local/bin" ]; then
        echo "export PATH=${NCU_DIR}:\$PATH" >> ~/.bashrc
        echo "已添加 ${NCU_DIR} 到 PATH"
    fi
fi

echo "========================================="
echo "ncu 安装完成"
echo "用法: ncu --set full -o profile_report ./your_program"
echo "========================================="
