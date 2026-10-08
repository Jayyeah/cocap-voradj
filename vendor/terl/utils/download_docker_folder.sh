#!/bin/bash

# 设置变量
SERVER="4090"
CONTAINER_NAME="pe-ch"
CONTAINER_FOLDER="/home/zhangheng/pursuit-evasion/experiment_data/exp_data_2024-10-21/training_data0__training_2024-10-15-15-37-44"
SERVER_PATH="/home/zhangheng/output.tar.gz"
LOCAL_PATH="$HOME/output.tar.gz"
EXTRACT_PATH="$HOME/extracted_folder"

# 函数：检查上一个命令是否成功
check_error() {
    if [ $? -ne 0 ]; then
        echo "错误：$1"
        exit 1
    fi
}

# 步骤1：在容器内压缩文件夹并保存到远程服务器
echo "正在压缩容器内的文件夹并保存到远程服务器..."
ssh $SERVER "docker exec $CONTAINER_NAME tar czf - '$CONTAINER_FOLDER' > $SERVER_PATH"
check_error "压缩文件夹失败"

echo "文件夹已压缩并保存到服务器的 $SERVER_PATH"

# 步骤2：从远程服务器下载压缩文件到本地
echo "正在从远程服务器下载压缩文件到本地..."
scp $SERVER:$SERVER_PATH "$LOCAL_PATH"
check_error "下载文件失败"

# 步骤3：解压缩文件
echo "正在解压缩文件..."
mkdir -p "$EXTRACT_PATH"
check_error "创建解压目录失败"

tar -xzf "$LOCAL_PATH" -C "$EXTRACT_PATH"
check_error "解压文件失败"

# 清理：删除本地压缩文件和远程服务器上的临时文件
rm "$LOCAL_PATH"
check_error "删除本地临时文件失败"

ssh $SERVER "rm $SERVER_PATH"
check_error "删除远程临时文件失败"

echo "文件夹已成功下载并解压缩到 $EXTRACT_PATH"