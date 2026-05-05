#!/bin/bash

# ==========================================
# 配置文件和参数
# ==========================================
PYTHON_SCRIPT="./src/scripts/local_push.py"  # 替换为你的 Python 脚本名
OUTPUT_JSON="./results/performance_experiment_results.json"

# 扰动比例设置 (对应 0.01%, 0.05%, 0.1%, 0.5%, 1%, 5%, 10%)
RATIOS=(0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1)
LABELS=("0.01%" "0.05%" "0.1%" "0.5%" "1%" "5%" "10%")
NUM_RATIOS=${#RATIOS[@]}

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 找不到 Python 脚本 '$PYTHON_SCRIPT'！请修改脚本中的 PYTHON_SCRIPT 路径。"
    exit 1
fi

# 初始化 JSON 数组
echo "[" > $OUTPUT_JSON

echo "开始执行增量式 PageRank 规模化实验..."
echo "结果将保存至: $OUTPUT_JSON"

# ==========================================
# 遍历不同扰动比例并执行
# ==========================================
for i in "${!RATIOS[@]}"; do
    ratio=${RATIOS[$i]}
    label=${LABELS[$i]}
    
    echo "--------------------------------------------------------"
    echo "▶ 正在运行扰动比例: $label (add_ratio=$ratio, drop_ratio=$ratio)"
    
    # 执行 Python 脚本并捕获输出
    output=$(python $PYTHON_SCRIPT --add_ratio $ratio --drop_ratio $ratio)
    
    # （可选）将输出打印到控制台，以便实时查看运行情况
    # echo "$output"
    
    # ------------------------------------------------------
    # 使用 grep 和 awk 从输出中提取关键信息
    # ------------------------------------------------------
    # 1. 提取 Cold Start 数据
    cold_time=$(echo "$output" | grep -A 2 "METHOD: Cold Start" | grep "Execution Time:" | awk '{print $3}')
    cold_iters=$(echo "$output" | grep -A 2 "METHOD: Cold Start" | grep "Iterations" | awk '{print $3}')
    
    # 2. 提取 Warm-Start 数据
    warm_time=$(echo "$output" | grep -A 2 "METHOD: Warm-Start" | grep "Execution Time:" | awk '{print $3}')
    warm_iters=$(echo "$output" | grep -A 2 "METHOD: Warm-Start" | grep "Iterations" | awk '{print $3}')
    warm_speedup=$(echo "$output" | grep "Warm-Start Speedup:" | awk '{print $4}' | tr -d 'x')
    
    # 3. 提取 Local Push 数据
    local_time=$(echo "$output" | grep -A 2 "METHOD: Local Push" | grep "Execution Time:" | awk '{print $3}')
    local_iters=$(echo "$output" | grep -A 2 "METHOD: Local Push" | grep "Iterations" | awk '{print $3}')
    local_speedup=$(echo "$output" | grep "local Push Speedup:" | awk '{print $5}' | tr -d 'x')

    # 为防止解析失败，提供默认值 0
    cold_time=${cold_time:-0}
    cold_iters=${cold_iters:-0}
    warm_time=${warm_time:-0}
    warm_iters=${warm_iters:-0}
    warm_speedup=${warm_speedup:-0}
    local_time=${local_time:-0}
    local_iters=${local_iters:-0}
    local_speedup=${local_speedup:-0}

    # ------------------------------------------------------
    # 构建当前比例的 JSON 对象
    # ------------------------------------------------------
    json_obj=$(cat <<EOF
  {
    "perturbation_percent": "$label",
    "perturbation_ratio": $ratio,
    "cold_start": {
      "iterations": $cold_iters,
      "execution_time_sec": $cold_time
    },
    "warm_start": {
      "iterations": $warm_iters,
      "execution_time_sec": $warm_time,
      "speedup": $warm_speedup
    },
    "local_push": {
      "iterations": $local_iters,
      "execution_time_sec": $local_time,
      "speedup": $local_speedup
    }
  }
EOF
)
    
    # 追加到 JSON 文件中
    echo "$json_obj" >> $OUTPUT_JSON
    
    # 如果不是最后一个元素，则添加逗号
    if [ $i -lt $((NUM_RATIOS - 1)) ]; then
        echo "," >> $OUTPUT_JSON
    fi
    
    echo "  ✔ 提取完成：Cold(${cold_iters}次), Warm(${warm_iters}次, ${warm_speedup}x), LocalPush(${local_iters}次, ${local_speedup}x)"
done

# 闭合 JSON 数组
echo "]" >> $OUTPUT_JSON

echo "--------------------------------------------------------"
echo "🎉 所有实验已完成！统计数据已写入 $OUTPUT_JSON 文件。"