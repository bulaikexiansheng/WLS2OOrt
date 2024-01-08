#!/bin/bash

# 运行Python文件
python Client.py &

# 获取上一个命令的进程ID（PID）
PYTHON_PID=$!
echo "$PYTHON_PID"

# 等待Python进程结束
wait $PYTHON_PID

# 终止Python进程
kill $PYTHON_PID
