# Progress records for incremental PageRank project

<center>Haopeng Zhang</center>

<center>Using dataset: [wiki-Vote from stanford](https://snap.stanford.edu/data/wiki-Vote.html).</center>

## Step 1: Theoretical Understanding of PageRank

> [!Note]
>
> TODO:
>
> 1. PageRank 及其数值算法基础
>
> - PageRank 原理与公式推导（Google 原论文、教材相关章节）
> - 幂法（Power Iteration）在 PageRank 中的应用
> - 稀疏矩阵存储与高效乘法（CSR/CSC 格式）
>
> 2. 图论与网络科学
>
> - 有向图的基本性质（强连通分量、悬挂节点等）
> - 真实网络的统计特性（如度分布、幂律分布、小世界现象）
> - 网络数据集的常见格式与解析方法（如 SNAP 格式）
>
> 3. 矩阵扰动理论与灵敏度分析
>
> - 矩阵扰动理论基础（如 Bauer-Fike 定理、谱半径扰动界）
> - PageRank 向量对转移矩阵扰动的敏感性分析
> - 残差与误差界的推导方法
>
> 4. 增量式/动态 PageRank 算法
>
> - 增量式 PageRank 的基本思想与主流算法（如 Online PageRank、Dynamic PageRank）
> - 残差驱动的增量迭代方法
> - 动态网络下的高效数据结构与实现技巧
>
> 5. 工程实现与实验设计
>
> - 大规模稀疏图的高效存储与处理（如 NetworkX、SciPy.sparse）
> - 性能分析与优化（如向量化、并行化、内存管理）
> - 实验设计与数据可视化（如对比实验、误差分析、可视化工具 matplotlib/seaborn）
>
> 6. 论文阅读与前沿进展
>
> - 经典 PageRank 相关论文（Google 原论文、增量 PageRank 相关文献）
> - 近期动态网络与增量算法的研究进展（可检索 arXiv、Google Scholar）
>
> ---
>
> **建议学习顺序**：
> PageRank 基础 → 图论与网络科学 → 矩阵扰动理论 → 增量式 PageRank → 工程实现 → 论文阅读与前沿进展
>

## Step 2: Environment Setup and Implementation of Static PageRank

```bash
conda create -n na_pr_env python=3.10
conda activate na_pr_env
pip install -r requirements.txt
```