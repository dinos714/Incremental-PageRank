# Extensional Experiments Plan

本文档列出了在原项目基础上可以继续拓展的 10 项工作。每一项都给出了具体思路、依托的现有代码接口、以及大致的实现路线。你不需要一次全部做完，可以按兴趣和难度挑选。

---

## 1. 逐节点 Gauss-Southwell Local Push

### 现状

当前 `src/scripts/local_push.py` 的第 9-19 行 `vectorized_local_push()` 实现的是**向量化批量推送**：每轮把所有节点的残差同时加到 x 上，再统一向邻居传播。这不是经典的 Gauss-Southwell 算法。

真正的 Gauss-Southwell 应该是：**每轮只选残差绝对值最大的那个节点**，单独推送它的残差，然后更新受影响的邻居残差，再选下一个最大残差节点。

### 为什么要做

- 逐节点推送的总计算量可能远小于批量推送（只处理真正受影响的局部区域）
- 这是原论文中的标准算法，对比向量化版本可以得出有意义的结论
- `progress.md` 第 291-303 行描述的正是逐节点版本，但代码并未实现

### 实现思路

新建文件 `src/scripts/gs_local_push.py`，核心函数：

```python
def gauss_southwell_local_push(G_new, D_new, f_new, x_old, n_nodes, p, tol=1e-9):
    # 1. 计算初始残差 r = A_new @ x_old - x_old
    Ax = p * G_new @ (D_new * x_old) + np.dot(f_new, x_old)
    r = Ax - x_old
    x = x_old.copy()
    iters = 0

    # 2. 循环：找出 |r| 最大的节点，推送它的残差
    while True:
        i = np.argmax(np.abs(r))        # 残差最大的节点
        if abs(r[i]) <= tol:
            break
        iters += 1

        # 推送：x[i] += r[i]
        x[i] += r[i]

        # 沿出边传播：r[j] += p * r[i] / out_deg[i]
        # 因为 G_new 是 CSR 格式，用 G_new[:, i] 取第 i 列的入边列表是不对的
        # G_new[rows, cols] 表示 cols -> rows，所以 i 的出边邻居 = G_new 第 i 列的非零行
        col_i = G_new[:, i]  # 这是一个 (n,) 的稀疏列切片
        if col_i.nnz > 0:
            # r[col_i.indices] += p * r[i] * D_new[i]   # D_new[i] = 1/out_deg[i]（对非悬挂节点）
            np.add.at(r, col_i.indices, p * r[i] * (col_i.data * D_new[i]))

        # 悬挂节点补偿：f_new @ r 的贡献分散到所有节点
        # 实际上推送时，r[i] 的 f_new 部分：每个节点收到 f_new[i] * r[i]
        # 但 f_new 本身就是 p 的函数，所以这里需要仔细处理
        # 简化版：r += f_new * r[i]  # 悬挂节点补偿
        r += f_new * r[i]

        r[i] = 0.0  # 清零

    x /= np.linalg.norm(x, 1)
    return x, iters
```

**关键点**：上述悬空节点补偿部分需要根据 `f_new` 的定义仔细推导。`f_new[j]` 的含义是：当冲浪者位于节点 j 时，通过随机跳转到达任意节点的概率。推送时 r[i] 应该乘以 `(1-p)/n` 分摊到所有节点（对非悬挂节点），或乘以 `1/n`（对悬挂节点）。实际上 f_new 已经包含这个信息，所以 `r += f_new * r[i]` 即可。

但这个加法会对所有 n 个节点做全量加，效率很差。更实际的做法是用一个标量 `total_compensation` 累积，最后统一加上。

### 实验设计

- 复用 `run_performance.sh` 的实验框架，把 GS Local Push 加入对比
- 关键对比：GS 的迭代次数 vs 向量化版本的迭代次数（GS 每轮只推一个节点，而向量化每轮推所有节点，所以迭代次数的含义不同，需要同时对比总运行时间和实际 FLOPS）

---

## 2. 外推加速技术（Aitken / Chebyshev）

### 现状

三种方法都使用最基础的迭代格式，没有任何加速技术。

### 为什么要做

当 $p$ 接近 1（例如 0.95）时，谱隙接近 0，幂法收敛极慢。Aitken 外推或 Chebyshev 加速可以显著减少迭代次数，尤其对 Cold Start 和 Warm-Start 有效。

### 实现思路：Aitken $\Delta^2$ 外推

在 `src/scripts/warm_start.py` 的 `power_method_with_init_vec()` 中插入外推逻辑。每 3 轮迭代用一次 Aitken 外推：

```python
def power_method_aitken(G, D, e, f, n_nodes, p, x_init=None, max_iter=500, tol=1e-9):
    if x_init is None:
        x = np.random.rand(n_nodes)
    else:
        x = x.copy()
    x /= np.linalg.norm(x, 1)

    x_prev2 = x.copy()  # k-2
    x_prev1 = x.copy()  # k-1

    for it in range(max_iter):
        x_new = p * G @ (D * x) + e * (f @ x)
        x_new /= np.linalg.norm(x_new, 1)

        # 每 3 轮尝试 Aitken 外推
        if it >= 2 and (it + 1) % 3 == 0:
            # 对每个分量分别外推（避免除零）
            epsilon = 1e-15
            denom = x_new - 2 * x_prev1 + x_prev2
            safe = np.abs(denom) > epsilon
            x_aitken = x_new.copy()
            x_aitken[safe] = x_new[safe] - (x_new[safe] - x_prev1[safe])**2 / denom[safe]
            x_aitken /= np.linalg.norm(x_aitken, 1)
            if np.linalg.norm(x_aitken - x_new, 1) < tol:
                return x_aitken, it + 1

        if np.linalg.norm(x_new - x, 1) < tol:
            return x_new, it + 1

        x_prev2 = x_prev1
        x_prev1 = x_new
        x = x_new
    return x, max_iter
```

### 实现思路：Chebyshev 多项式加速

幂法的等价形式是将残差按矩阵特征值展开。Chebyshev 加速通过构造多项式来压制非主特征值分量。对于 PageRank 问题，已知第二特征值 $\lambda_2 = p$（在无 dangling node 的情况下），可以利用这个信息。

简化实现：对于函数 $f(A) = (I - p\bar{M})^{-1}$，可以用 Chebyshev 迭代直接逼近 PageRank 向量而不用幂法。具体可以参考 `references/langville_meyer_2006.pdf` 的第 9 章。

### 实验设计

- 新建 `src/scripts/accelerated_methods.py`
- 在 p=0.95 时对比：Cold Start vs Cold+Aitken vs Cold+Chebyshev vs Warm+Aitken 的迭代次数和时间
- 加入 `run_sensitivity.sh` 的 p-sensitivity 实验

---

## 3. 节点增删支持

### 现状

`perturb_graph()`（`warm_start.py` 第 25-39 行）只处理边的增删，节点数不变。

### 为什么要做

实际动态网络中，新网页出现和旧网页消失都很常见。节点增删需要扩展矩阵维度，比纯边增删更复杂。

### 实现思路

在 `warm_start.py` 中新增 `perturb_graph_with_nodes()`：

```python
def perturb_graph_with_nodes(edges, n_nodes, add_node_ratio=0.01, drop_node_ratio=0.01,
                              add_edge_ratio=0.001, drop_edge_ratio=0.001):
    """
    同时处理节点和边的增删。
    - add_node_ratio: 新增节点数 / 原节点数
    - drop_node_ratio: 删除节点数 / 原节点数
    """
    # 1. 随机删除节点
    n_drop = int(n_nodes * drop_node_ratio)
    drop_nodes = set(np.random.choice(n_nodes, n_drop, replace=False))

    # 过滤掉涉及被删节点的边
    mask = ~(np.isin(edges[:, 0], list(drop_nodes)) | np.isin(edges[:, 1], list(drop_nodes)))
    edges_kept = edges[mask]

    # 重建节点编号映射（被删节点移除后，重新编号为 0..n_new-1）
    ...

    # 2. 新增节点（创建新边）
    ...

    return edges_new, n_nodes_new, stats
```

### 关键的坑

- 节点删除后需要重新 compact 编号，否则稀疏矩阵会变大且稀疏
- Warm-Start 的初始向量也需要相应裁剪/扩展（删掉的节点去掉，新增节点用均匀值 $1/n$ 填充）
- 计算 $\Delta \bar{M}$ 的 L1 范数时需要处理维度变化

### 实验设计

- 单独写一个 `src/scripts/node_perturbation.py`
- 控制变量：固定边扰动为 0，变化节点增删比例（1%, 5%, 10%）
- 观察 Warm-Start 和 Local Push 的加速比变化

---

## 4. 谱分析与收敛性研究

### 现状

`progress.md` 提到了谱间隙（spectral gap）是收敛速度的关键，并在第 454-457 行解释了 10% 扰动时迭代激增的原因。但没有做定量分析。

### 为什么要做

这是数值分析课程的核心内容——理解算法收敛性的数学本质。可以定量解释：为什么 p=0.95 时收敛慢？为什么 10% 扰动时所有方法都暴增到 80 次迭代？Local Push 的收敛率与谱间隙的关系是什么？

### 实现思路

#### 4a. 计算谱间隙

PageRank 转移矩阵 $A = p\bar{M} + (1-p)\vec{e}\vec{v}^T$ 的特征值为：$\lambda_1=1, |\lambda_2| \le p, |\lambda_3| \le p, \dots$

谱间隙 = $1 - |\lambda_2|$。理论上 $|\lambda_2| \le p$，但实际中常常更小。

用 `scipy.sparse.linalg.eigs` 求前几个特征值（对中等规模图像可行，Wiki-Vote 有 7115 节点，可以算）：

```python
from scipy.sparse.linalg import eigs

def compute_spectral_gap(A_bar, p):
    """
    A_bar = p*M_bar + (1-p)*e*v^T  (n x n 列随机矩阵的 CSR 形式)
    求 |lambda_2|，返回 1 - |lambda_2|
    """
    vals, _ = eigs(A_bar, k=3, which='LM')
    vals = sorted(np.abs(vals), reverse=True)
    return 1 - vals[1]   # vals[0] ≈ 1
```

#### 4b. 幂法收敛率 vs 谱间隙

幂法的收敛率：$||x^{(k)} - x^*|| \approx O(|\lambda_2/\lambda_1|^k)$

做实验：给定不同的 p 值和不同的扰动比例，分别测量：
- 实际收敛所需迭代次数
- 理论预测 $k \approx \log(\text{tol}) / \log(|\lambda_2|)$

#### 4c. Davis-Kahan 定理给出更紧的摄动界

当前使用的上界 $\frac{p}{1-p}||\Delta \bar{M}||_1$ 过于保守。Davis-Kahan 定理给出特征向量摄动的界：

$$||\Delta \vec{x}|| \le \frac{||\Delta A||}{\delta}$$

其中 $\delta$ 是 $A$ 的特征值隔离度（谱间隙）。因为 $\delta = 1 - |\lambda_2|$，在 p 接近 1 时 $\delta$ 很小，意味着同样的矩阵扰动会导致更大的 PageRank 变化。这可以解释 p=0.95 时加速比下降的现象。

```python
def davis_kahan_bound(delta_A_norm, spectral_gap):
    return delta_A_norm / spectral_gap
```

### 实验设计

- 新建 `src/scripts/spectral_analysis.py`
- 用 Wiki-Vote 和 Textbook-Example 两个数据集（小图可以精确计算全部特征值）
- 画图：（1）p vs 谱间隙曲线；（2）扰动比例 vs 实际迭代次数 vs 理论预测迭代次数
- 在 `results/` 下新增 `spectral_analysis.png`

---

## 5. 边增加 vs 边删除的分别分析

### 现状

`perturb_graph()` 同时做 add 和 drop，用同一个 ratio。`matrix_perturbation.py` 虽然有 `--variable add_ratio` 和 `--variable drop_ratio` 参数，但只做了单变量变化的实验，没有交叉对比。

### 为什么要做

增加边和删除边对转移矩阵的影响是完全不对称的。删除边可能导致新的 dangling node 出现，增加边不会。这个问题本身在 PageRank 理论中也有意义。

### 实现思路

扩展 `matrix_perturbation.py` 的 `run_experiments_and_plot()`，做 2D 热力图：

- x 轴：add_ratio（0.001, 0.005, 0.01, 0.05, 0.1）
- y 轴：drop_ratio（同上）
- 颜色：$||\Delta \vec{x}||_1$ 或加速比

```python
def run_2d_perturbation_experiment(edges, n_nodes, p, ratios, output_path):
    results = np.zeros((len(ratios), len(ratios)))
    for i, add_r in enumerate(ratios):
        for j, drop_r in enumerate(ratios):
            dx, _, _, _ = simulate_perturbation(edges, n_nodes, p, add_r, drop_r)
            results[i, j] = dx
    # 画热力图
    plt.imshow(results, ...)
```

也可以对比：相同总扰动下（add+drop=常数），纯 add 和纯 drop 哪个对 PageRank 影响更大？

### 实验设计

- 直接在 `matrix_perturbation.py` 中加一个 `--mode 2d` 参数
- 生成热力图保存到 `results/perturbation_2d_heatmap.png`

---

## 6. Top-k 排序稳定性分析

### 现状

当前只检查了 L1 范数误差（`||x_cold - x_warm||_1`）和理论界（`||Δx||_1`）。但实际用户只关心**网页排名变了没有**，不关心 L1 范数。

### 为什么要做

- L1 误差小不等于排序稳。有可能 L1 误差很小，但 Top-10 的排序发生了翻天覆地的变化。
- 这是从"数值分析"走向"实际应用"的自然延伸。

### 度量指标

1. **Kendall $\tau$**：衡量两个排序的一致性（-1 到 1，1 表示完全一致）
2. **Spearman 秩相关系数**：类似于 Pearson 相关系数但基于秩
3. **Top-k Jaccard 相似度**：前 k 名中有多少节点是相同的（不考虑内部顺序）
4. **Top-k 重叠率 + 位移**：前 k 名中相同节点的最大排名位移

### 实现

```python
from scipy.stats import kendalltau, spearmanr

def ranking_stability_metrics(x_old, x_new, top_k_list=[10, 50, 100]):
    metrics = {}
    # 全量排序
    rank_old = np.argsort(x_old)[::-1]
    rank_new = np.argsort(x_new)[::-1]

    # Kendall tau（对所有节点比较慢，可以采样或用 scipy）
    tau, _ = kendalltau(x_old, x_new)
    rho, _ = spearmanr(x_old, x_new)

    metrics['kendall_tau'] = tau
    metrics['spearman_rho'] = rho

    for k in top_k_list:
        top_old = set(rank_old[:k])
        top_new = set(rank_new[:k])
        metrics[f'jaccard_top{k}'] = len(top_old & top_new) / len(top_old | top_new)

    return metrics
```

### 实验设计

- 新建 `src/scripts/ranking_stability.py`
- 固定几种扰动比例（0.1%, 0.5%, 1%, 5%, 10%），分别计算排序稳定性指标
- 画折线图：x 轴 = 扰动比例，y 轴 = Kendall $\tau$ / Top-10 Jaccard
- 同时对比 L1 范数变化 vs 排序变化，验证是否 L1 误差能很好地代理排序稳定性

---

## 7. Personalized PageRank 的增量更新

### 现状

所有实现都使用全局均匀 teleportation 向量 $\vec{v} = \vec{e}/n$。即用户随机跳转时等概率访问任何网页。

### 为什么要做

- 实际搜索引擎中，个性化 PageRank 是标配（用户有偏好网页集合）
- 增量更新在个性化场景下更有意义：用户偏好变化 + 网络结构变化 叠加

### 实现思路

修改 `build_components()` 和所有幂法函数，接受一个可选的 teleportation 向量参数 `v`：

```python
def build_components_personalized(edges, n_nodes, p, preference_nodes=None):
    """
    preference_nodes: 用户偏好节点列表，跳转时以更高概率回到这些节点
    """
    G, D, e, f, _ = build_components(edges, n_nodes, p)
    v = np.zeros(n_nodes)
    if preference_nodes is not None:
        v[preference_nodes] = 1.0 / len(preference_nodes)
    else:
        v = np.ones(n_nodes) / n_nodes
    return G, D, e, f, v
```

幂法公式变为 $x = pGDx + \vec{e} \cdot (f^T x)$（$f$ 中已包含 teleportation 分布）

### 增量更新场景

假设网络结构变化（边增删）同时用户偏好也变化（比如用户收藏了新网页，移除了旧网页）。可以设计两种更新策略：
- **策略 A**：先更新网络，再更新偏好（顺序）
- **策略 B**：同时更新（联合增量）

对比两种策略的效率差异。

### 实验设计

- 新建 `src/scripts/personalized_pr.py`
- 随机选 5-10 个节点作为偏好节点
- 同时扰动图和偏好，对比 Cold/Warm/LocalPush 的表现

---

## 8. 多数据集扩展

### 现状

只用了两个数据集：Textbook-Example（6 节点）和 Wiki-Vote（7115 节点）。

### 需要补充的数据集

从 [Stanford SNAP](https://snap.stanford.edu/data/) 下载，放入 `data/` 目录：

| 数据集 | 节点数 | 边数 | 类型 | 特点 |
|--------|--------|------|------|------|
| ego-Facebook | 4,039 | 88,234 | 无向社交网络 | 高聚类系数 |
| Web-Google | 875,713 | 5,105,039 | 有向网页图 | 大规模 |
| p2p-Gnutella | 62,586 | 147,892 | 有向 P2P 网络 | 中等规模、高动态 |
| soc-Epinions1 | 75,879 | 508,837 | 有向信任网络 | 社会网络 |
| email-EuAll | 265,214 | 420,045 | 有向邮件网络 | 大规模稀疏 |
| cit-HepPh | 34,546 | 421,578 | 有向引用网络 | 学术引用 |

### 下载方式

```bash
# 以 Wiki-Vote 为例（已下载），其他类似
wget https://snap.stanford.edu/data/facebook_combined.txt.gz
gunzip facebook_combined.txt.gz
mv facebook_combined.txt data/facebook.txt
```

### 实现

- 写一个 `data/download_datasets.sh` 自动下载脚本
- 写一个 `src/scripts/benchmark_datasets.py`：对所有数据集运行 Cold/Warm/LocalPush，记录迭代次数和时间
- 分析：图规模、密度、平均出度等结构参数对增量算法加速比的影响

### 注意事项

- `load_graph()` 函数需要兼容无向图（无向边需要当作两条有向边处理，或者下载有向版本）
- 超过 10 万节点的图，`eigs` 谱分析会很慢，需要降维采样或用幂法迭代求 $\lambda_2$

---

## 9. 蒙特卡洛 / 随机游走方法对比

### 现状

所有方法都基于幂法/残差推送。没有任何基于采样的方法。

### 为什么要做

蒙特卡洛方法是一种完全不同的 PageRank 计算范式：通过模拟大量随机游走，统计各节点的访问频率。它的优势在于：
- 不需要存储和操作整个转移矩阵
- 可以增量更新（只需在新的图上重走部分游走路径）
- 天然支持个性化 PageRank

### 实现思路

```python
def mc_pagerank(edges, n_nodes, p, n_walks=100000, walk_length=20):
    """
    从每个节点出发，模拟随机游走，统计访问次数。
    """
    counts = np.zeros(n_nodes)
    out_deg = np.bincount(edges[:, 0], minlength=n_nodes)

    for start_node in range(n_nodes):
        for _ in range(n_walks // n_nodes):
            current = start_node
            for _ in range(walk_length):
                counts[current] += 1
                if np.random.random() > p:
                    # 随机跳转
                    current = np.random.randint(n_nodes)
                elif out_deg[current] > 0:
                    # 沿出边随机走一步
                    neighbors = edges[edges[:, 0] == current][:, 1]
                    current = np.random.choice(neighbors)
                else:
                    # 悬挂节点 -> 随机跳转
                    current = np.random.randint(n_nodes)

    return counts / counts.sum()
```

增量版本：

```python
def mc_pagerank_incremental(edges_old, edges_new, n_nodes_old, n_nodes_new, p, n_walks=100000):
    """
    只重新模拟受到影响的那部分游走路径。
    找出拓扑变化的局部区域，只在该区域内重走游走。
    """
    ...
```

### 实验设计

- 新建 `src/scripts/mc_pagerank.py`
- 对比：MC 方法的精度 vs 时间 vs 幂法（以幂法结果为 ground truth）
- 对比：MC 增量更新 vs MC 全局重算

---

## 10. 自适应局部更新策略

### 现状

Local Push 是对所有残差不为零的节点做推送。但很多节点的残差非常小，推送它们收益甚微。

### 为什么要做

设置一个阈值 $\theta > \text{tol}$：只推送 $|r_i| > \theta$ 的节点。当最大残差降到 $\theta$ 以下后，再转向全局推送。这样可以减少大量低收益的推送操作。

### 实现思路

在 GS Local Push 的基础上，改 main loop：

```python
def adaptive_local_push(G_new, D_new, f_new, x_old, n_nodes, p, tol=1e-9, theta=None):
    if theta is None:
        theta = np.sqrt(tol)  # 启发式阈值

    # Phase 1: 只推送大残差节点
    while True:
        i = np.argmax(np.abs(r))
        if abs(r[i]) <= theta:
            break
        # 单点推送（同 GS）
        ...

    # Phase 2: 对剩余小残差，回退到向量化批量推送
    x, extra_iters = vectorized_local_push(G_new, D_new, f_new, x, n_nodes, p, tol=tol)
    return x, iters_phase1 + extra_iters
```

### 实验设计

- 直接在 `gs_local_push.py` 中添加 `adaptive_local_push()`
- 对比：GS vs Adaptive vs Vectorized 的迭代次数、时间和精度
- 尝试不同 $\theta$ 值（`tol*10`, `tol*100`, `tol*1000`）的效果

---

## 全局建议

### 优先级排序（按投入产出比）

| 优先级 | 工作 | 预计工作量 | 原因 |
|--------|------|-----------|------|
| ★★★ | 1. GS Local Push | 半天 | 算法完整性的关键缺口 |
| ★★★ | 6. Top-k 排序稳定性 | 半天 | 实际意义强，实现简单 |
| ★★★ | 5. 边增删分别分析 | 2-3 小时 | 在已有代码上扩展即可 |
| ★★☆ | 4. 谱分析 | 1 天 | 理论深度好，但需要熟悉 scipy.sparse.linalg |
| ★★☆ | 8. 多数据集 | 半天（主要是下载和数据清洗） | 增强实验说服力 |
| ★★☆ | 2. 外推加速 | 1 天 | 实现有一定复杂度（尤其 Chebyshev） |
| ★☆☆ | 3. 节点增删 | 1-2 天 | 代码改动面大，需要仔细处理编号重映射 |
| ★☆☆ | 7. Personalized PR | 半天 | 改动小但实验设计需要想清楚 |
| ★☆☆ | 10. 自适应阈值 | 半天 | 需要 GS 先做出来 |
| ★☆☆ | 9. 蒙特卡洛 | 1 天 | 完全不同的范式，需要较多编码 |

### 代码组织建议

每项新工作建议在 `src/scripts/` 下新建独立脚本，复用 `static_textook_implementation_power_method.py` 中的 `load_graph()`、`build_components()` 等基础函数。不要在已有脚本上大改，保持向后兼容。

实验结果统一输出到 `results/` 目录，可视化统一在 `src/notebooks/notes.ipynb` 中追加新的 cell。

### 如果时间有限，只做三件事

1. **GS Local Push**（补全算法完整性）
2. **Top-k 排序稳定性**（最有实际说服力）
3. **多数据集 + 边增删分别分析**（实验的广度和深度）
