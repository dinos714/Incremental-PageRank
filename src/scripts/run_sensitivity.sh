#!/bin/bash

# ==========================================
# 配置文件和参数
# ==========================================
PYTHON_SCRIPT="./src/scripts/local_push.py"  # 替换为你的 Python 脚本名
OUTPUT_JSON="./results/sensitivity_experiment_results.json"

# 固定参数 (选用 0.1% 的微小扰动以保证增量算法处于优势区间)
FIXED_RATIO=0.001
FIXED_P=0.85
FIXED_TOL=1e-9

# 需要遍历的超参数数组
P_VALUES=(0.70 0.85 0.90 0.95)
TOL_VALUES=(1e-6 1e-9 1e-12)

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 找不到 Python 脚本 '$PYTHON_SCRIPT'！"
    exit 1
fi

# 初始化 JSON
echo "{" > $OUTPUT_JSON

# ==========================================
# 实验 1: 阻尼系数 p 的敏感性分析
# ==========================================
echo "开始实验 1/2: 阻尼系数 p 的敏感性分析 (固定 tol=$FIXED_TOL, ratio=$FIXED_RATIO)"
echo "  \"p_sensitivity\":[" >> $OUTPUT_JSON

for i in "${!P_VALUES[@]}"; do
    p=${P_VALUES[$i]}
    echo "  ▶ 正在运行 p = $p"
    
    # 执行脚本
    output=$(python $PYTHON_SCRIPT --p $p --tol $FIXED_TOL --add_ratio $FIXED_RATIO --drop_ratio $FIXED_RATIO)
    
    # 解析输出 (匹配最新的 Python 脚本输出格式)
    cold_time=$(echo "$output" | grep -A 2 "METHOD: Cold Start" | grep "Execution Time:" | awk '{print $3}')
    cold_iters=$(echo "$output" | grep -A 2 "METHOD: Cold Start" | grep "Iterations" | awk '{print $3}')
    
    warm_time=$(echo "$output" | grep -A 2 "METHOD: Warm-Start" | grep "Execution Time:" | awk '{print $3}')
    warm_iters=$(echo "$output" | grep -A 2 "METHOD: Warm-Start" | grep "Iterations" | awk '{print $3}')
    warm_speedup=$(echo "$output" | grep "Warm-Start Speedup:" | awk '{print $4}' | tr -d 'x')
    
    local_time=$(echo "$output" | grep -A 2 "METHOD: Local Push" | grep "Execution Time:" | awk '{print $3}')
    local_iters=$(echo "$output" | grep -A 2 "METHOD: Local Push" | grep "Iterations" | awk '{print $3}')
    local_speedup=$(echo "$output" | grep "local Push Speedup:" | awk '{print $5}' | tr -d 'x')

    # 防止空值
    cold_time=${cold_time:-0}; cold_iters=${cold_iters:-0}
    warm_time=${warm_time:-0}; warm_iters=${warm_iters:-0}; warm_speedup=${warm_speedup:-0}
    local_time=${local_time:-0}; local_iters=${local_iters:-0}; local_speedup=${local_speedup:-0}

    # 写入 JSON
    json_obj=$(cat <<EOF
    {
      "p": $p,
      "tol": $FIXED_TOL,
      "cold_start": { "iterations": $cold_iters, "execution_time_sec": $cold_time },
      "warm_start": { "iterations": $warm_iters, "execution_time_sec": $warm_time, "speedup": $warm_speedup },
      "local_push": { "iterations": $local_iters, "execution_time_sec": $local_time, "speedup": $local_speedup }
    }
EOF
)
    echo "$json_obj" >> $OUTPUT_JSON
    if [ $i -lt $((${#P_VALUES[@]} - 1)) ]; then echo "    ," >> $OUTPUT_JSON; fi
    echo "    ✔ 完成: Cold(${cold_iters}次), Warm(${warm_iters}次), LocalPush(${local_iters}次)"
done

echo "  ]," >> $OUTPUT_JSON
echo "--------------------------------------------------------"

# ==========================================
# 实验 2: 容忍度 tol 的敏感性分析
# ==========================================
echo "开始实验 2/2: 收敛容忍度 tol 的敏感性分析 (固定 p=$FIXED_P, ratio=$FIXED_RATIO)"
echo "  \"tol_sensitivity\": [" >> $OUTPUT_JSON

for i in "${!TOL_VALUES[@]}"; do
    tol=${TOL_VALUES[$i]}
    echo "  ▶ 正在运行 tol = $tol"
    
    # 执行脚本 (注意传入科学计数法 tol 时 bash 与 python 的兼容)
    output=$(python $PYTHON_SCRIPT --p $FIXED_P --tol $tol --add_ratio $FIXED_RATIO --drop_ratio $FIXED_RATIO)
    
    cold_time=$(echo "$output" | grep -A 2 "METHOD: Cold Start" | grep "Execution Time:" | awk '{print $3}')
    cold_iters=$(echo "$output" | grep -A 2 "METHOD: Cold Start" | grep "Iterations" | awk '{print $3}')
    
    warm_time=$(echo "$output" | grep -A 2 "METHOD: Warm-Start" | grep "Execution Time:" | awk '{print $3}')
    warm_iters=$(echo "$output" | grep -A 2 "METHOD: Warm-Start" | grep "Iterations" | awk '{print $3}')
    warm_speedup=$(echo "$output" | grep "Warm-Start Speedup:" | awk '{print $4}' | tr -d 'x')
    
    local_time=$(echo "$output" | grep -A 2 "METHOD: Local Push" | grep "Execution Time:" | awk '{print $3}')
    local_iters=$(echo "$output" | grep -A 2 "METHOD: Local Push" | grep "Iterations" | awk '{print $3}')
    local_speedup=$(echo "$output" | grep "local Push Speedup:" | awk '{print $5}' | tr -d 'x')

    cold_time=${cold_time:-0}; cold_iters=${cold_iters:-0}
    warm_time=${warm_time:-0}; warm_iters=${warm_iters:-0}; warm_speedup=${warm_speedup:-0}
    local_time=${local_time:-0}; local_iters=${local_iters:-0}; local_speedup=${local_speedup:-0}

    json_obj=$(cat <<EOF
    {
      "p": $FIXED_P,
      "tol_str": "$tol",
      "cold_start": { "iterations": $cold_iters, "execution_time_sec": $cold_time },
      "warm_start": { "iterations": $warm_iters, "execution_time_sec": $warm_time, "speedup": $warm_speedup },
      "local_push": { "iterations": $local_iters, "execution_time_sec": $local_time, "speedup": $local_speedup }
    }
EOF
)
    echo "$json_obj" >> $OUTPUT_JSON
    if [ $i -lt $((${#TOL_VALUES[@]} - 1)) ]; then echo "    ," >> $OUTPUT_JSON; fi
    echo "    ✔ 完成: Cold(${cold_iters}次), Warm(${warm_iters}次), LocalPush(${local_iters}次)"
done

echo "  ]" >> $OUTPUT_JSON
echo "}" >> $OUTPUT_JSON

echo "--------------------------------------------------------"
echo "🎉 敏感性分析实验完毕！数据已保存至: $OUTPUT_JSON"