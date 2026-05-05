# 动态网络下的 PageRank 增量算法与扰动分析

## 项目简介

本项目探索在网络结构发生微小变动时，如何利用矩阵扰动理论来量化网络变化对 PageRank 排名的影响，并且设计并实现高效的增量式更新算法。

## 理论背景

### PageRank 模型

PageRank 基于随机冲浪模型：用户以概率 $p$ 沿当前页面链接跳转，以概率 $1-p$ 随机访问新页面。转移矩阵 $A$ 定义为：

$$A = pGD + \vec{e}\vec{f}^T$$

其中 $G$ 为邻接矩阵， $D$ 为出度倒数对角矩阵， $\vec{f}$ 处理悬挂节点。

### 扰动理论界

当网络拓扑发生扰动 $\Delta \bar{M}$ 时，PageRank 向量变化满足：

$$\|\Delta \vec{x}\|_1 \leq \frac{p}{1-p} \|\Delta \bar{M}\|_1$$

## 项目结构

```
.
├── src/
│   ├── scripts/
│   │   ├── static_textook_implementation_power_method.py   # 静态PageRank实现
│   │   ├── warm_start.py                                   # 预热启动算法
│   │   ├── local_push.py                                   # 本地推送算法
│   │   ├── matrix_perturbation.py                          # 扰动分析脚本
│   │   ├── run_performance.sh                              # 性能实验脚本
│   │   └── run_sensitivity.sh                              # 敏感性分析脚本
│   └── notebooks/
│       └── notes.ipynb
├── data/
│   ├── Textbook-Example.txt    # 小规模示例数据（6节点，9边）
│   └── Wiki-Vote.txt           # Wikipedia投票网络（7115节点，103689边）
├── results/                    # 实验结果图表
├── references/                 # 参考资料
├── proposal.md                 # 项目提案
├── progress.md                 # 进度记录
└── requirements.txt            # Python依赖
```

## 增量算法

本项目实现了三种 PageRank 计算方法：

1. **Cold Start（冷启动）**: 从随机向量开始迭代。
2. **Warm-Start（预热启动）**: 使用上一次计算结果作为初始向量。
3. **Local Push（局部推送）**: 基于残差的增量更新算法。

## 环境配置

```bash
conda create -n na_pr_env python=3.10
conda activate na_pr_env
pip install -r requirements.txt
```

## 使用方法

### 静态 PageRank 计算

```bash
python src/scripts/static_textook_implementation_power_method.py \
    --data_file_path ./data/Wiki-Vote.txt \
    --p 0.85
```

### 增量更新对比

```bash
python src/scripts/local_push.py \
    --data_file_path ./data/Wiki-Vote.txt \
    --add_ratio 0.001 \
    --drop_ratio 0.001
```

### 矩阵扰动分析

```bash
python src/scripts/matrix_perturbation.py \
    --data_file_path ./data/Wiki-Vote.txt \
    --run_experiments \
    --variable drop_ratio
```

### 运行全部性能实验

```bash
bash src/scripts/run_performance.sh
bash src/scripts/run_sensitivity.sh
```

## 实验结果

实验结果表明：
- Warm-Start 和 Local Push 在小规模扰动下显著优于 Cold Start
- 当扰动比例较小时（<1%），增量算法的加速比可达数倍
- 扰动界公式验证了理论分析与实际结果的一致性

## 数据说明

- `Wiki-Vote.txt`: 来自 Stanford SNAP 的 Wikipedia 管理员投票网络
- 格式: 每行包含源节点 ID 和目标节点 ID，以 tab 分隔
- 以 `#` 开头的行为注释