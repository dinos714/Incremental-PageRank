


在上一阶段中，我们通过矩阵扰动理论和实验发现了一个关键现象：**尽管理论上界可能很大，但在实际的幂律网络中，拓扑的微小改变（如添加/删除少量边）只引起了 PageRank 向量的极小变化。**

这正是**增量式 PageRank（Incremental PageRank）**算法的立足之本。既然只有局部节点的概率分布受到了影响，我们完全不需要像静态计算那样从头开始全局迭代，而是可以**“哪里变化了，就只更新哪里”**。

结合你在 Step 1 中非常严谨的列随机矩阵 $\bar{M}$ 推导，这里为你提供增量式 PageRank 的理论推导与算法框架，你可以将其直接填补进 `progress.md` 的 **Step 3** 中。

---

### 增量式 PageRank 的理论基石：残差驱动与局部更新

传统的幂法（Power Method）基于公式 $\vec{x}^{(k+1)} = p\bar{M}\vec{x}^{(k)} + (1-p)\vec{v}$ 进行全局更新。而在增量算法中，我们引入**残差向量（Residual Vector） $\vec{r}$** 的概念。

假设在旧图 $G$ 上，PageRank 向量已经收敛为 $\vec{x}$，此时系统处于平衡状态，残差 $\vec{r} = \vec{0}$。
当图结构发生微小变化（生成新图 $G'$），转移矩阵从 $\bar{M}$ 变为 $\bar{M}'$，增量矩阵为 $\Delta \bar{M} = \bar{M}' - \bar{M}$。

在这个瞬间，原本的 $\vec{x}$ 尚未改变，但由于网络的“管道”变了，原本的概率分布不再平衡，系统产生了新的**初始残差**：
$$ \Delta \vec{r} = p \Delta \bar{M} \vec{x} $$

**最绝妙的地方在于 $\Delta \bar{M}$ 的极度稀疏性：**
假设我们仅仅在节点 $u$ 和 $v$ 之间**添加了一条边 $u \rightarrow v$**，这意味着**只有 $\bar{M}$ 的第 $u$ 列发生了变化！**
因此，产生的新残差仅仅与节点 $u$ 的旧 PageRank 值 $x_u$ 有关：
$$ \Delta \vec{r} = p \cdot x_u \cdot \Delta \bar{M}[:, u] $$

#### 对 $\Delta \bar{M}[:, u]$ 的严密分类讨论（结合悬挂节点）：

1. **情况 A：节点 $u$ 原本不是悬挂节点（出度 $d_u > 0 \rightarrow d_u + 1$）**
   原本 $u$ 将概率均匀分给 $d_u$ 个邻居，现在分给 $d_u + 1$ 个邻居。
   - 对 $u$ 的**旧邻居 $w$**：概率减少了，$\Delta \bar{M}[w, u] = \frac{1}{d_u+1} - \frac{1}{d_u} = \frac{-1}{d_u(d_u+1)}$
   - 对 $u$ 的**新邻居 $v$**：获得了新概率，$\Delta \bar{M}[v, u] = \frac{1}{d_u+1}$
   *(只有这 $d_u+1$ 个节点产生了残差，其余所有节点的残差仍为 0)*

2. **情况 B：节点 $u$ 原本是悬挂节点（出度 $d_u = 0 \rightarrow 1$）**
   原本 $u$ 是悬挂节点，它的概率通过向量 $\vec{d}$ 均匀补偿给全局每个节点 $\frac{1}{n}$。现在它连向了 $v$，不再是悬挂节点。
   - 对**新邻居 $v$**：获得了独占的概率，$\Delta \bar{M}[v, u] = 1 - \frac{1}{n}$
   - 对**图中除 $v$ 外的所有节点 $w$**：失去了补偿概率，$\Delta \bar{M}[w, u] = -\frac{1}{n}$
   *(这是增量计算中最耗时的一步，但实际网络中单次加边遇到悬挂节点的概率有限)*

---

### 增量式算法核心：Local Push（局部推送）

计算出初始残差 $\vec{r}$ 后，我们就进入增量迭代过程（如经典算法 Gauss-Southwell 或 Local Push）。我们不再进行全局矩阵乘法，而是**只处理残差大于容忍度 $\epsilon$ 的节点**。

**Local Push 算法逻辑：**
1. 维护一个队列或优先队列，存入所有 $|r_i| > \epsilon$ 的节点。
2. 当队列不为空时，弹出一个节点 $i$：
   - 将它的残差累加到它的真实 PageRank 值中：$x_i \leftarrow x_i + r_i$
   - 将它的残差沿着出边**推送（Push）**给它的邻居 $j$：$r_j \leftarrow r_j + p \frac{r_i}{d_i}$
   - 节点 $i$ 的残差清零：$r_i \leftarrow 0$
3. 循环直到所有节点的残差都在容忍度之下。

*(注：实际工程中为防正负残差抵消导致震荡，也可只更新 $x$，这在具体实现中会有细微变形。最简单的工程实现方案是 **Warm-Start（热启动法）**)*

---

### 为 `progress.md` 准备的补充文本

你可以将以下内容添加到你的 `progress.md` 中 **Step 3** 的位置：

```markdown
## Step3: Implementation of Incremental PageRank Algorithm

静态 PageRank 的幂法需要在每一步对整个大规模矩阵进行全量计算。然而通过 Step 1 矩阵扰动理论的推导与 Step 2 的实验证实：即使理论扰动上界很高，在实际网络中，拓扑的小幅变动（例如边的新增或删除）对全局平稳分布 $\vec{x}$ 的影响是极小的。绝大多数节点的重要度不变，变化仅局限在扰动发生的局部。

因此，增量式 PageRank（Incremental PageRank）的核心思想是：**复用旧图的收敛状态，仅对残差进行局部更新。**

#### **3.1 理论框架：残差驱动机制 (Residual-Driven Mechanism)**

设旧图的转移矩阵为 $\bar{M}$，已收敛的 PageRank 向量为 $\vec{x}$。当图结构发生变化生成新图 $\bar{M}'$ 时，增量矩阵为 $\Delta \bar{M} = \bar{M}' - \bar{M}$。
由于此时仍采用旧的 $\vec{x}$，网络失去了马尔可夫平衡，产生了**初始残差（Initial Residual）**：
$$ \Delta \vec{r} = p \Delta \bar{M} \vec{x} $$

**局部性分析：**
假设网络中仅新增了一条边 $(u, v)$，则只有矩阵的第 $u$ 列发生了变化（即 $\Delta \bar{M}$ 只有第 $u$ 列非零）。初始残差可以精确表达为：
$$ \Delta \vec{r} = p \cdot x_u \cdot \Delta \bar{M}[:, u] $$

结合我们在 Step 1 中关于悬挂节点补偿的理论，向量 $\Delta \bar{M}[:, u]$ 的分布严格取决于 $u$ 的状态：
1. **$u$ 为非悬挂节点 ($d_u > 0$)**：
   原有邻居 $w$ 的残差变为 $r_w = -p \cdot x_u \frac{1}{d_u(d_u+1)}$。
   新邻居 $v$ 的残差变为 $r_v = p \cdot x_u \frac{1}{d_u+1}$。
   其余节点的残差全部为 0。
2. **$u$ 原本为悬挂节点 ($d_u = 0$)**：
   $u$ 不再向全局广播均匀概率，其所有出度集中于 $v$。
   新邻居 $v$ 获得残差 $r_v = p \cdot x_u (1 - \frac{1}{n})$。
   其余所有节点失去悬挂补偿，获得残差 $r_w = -p \cdot x_u \frac{1}{n}$。

#### **3.2 算法实现路径 (Implementation Paths)**

在工程实现上，基于上述理论，我们可以落地两种主流的增量策略：

1. **热启动的幂法 (Warm-Start Power Method)**
   将旧的收敛向量 $\vec{x}_{old}$ 作为新图求特征向量的迭代初始值 $\vec{x}^{(0)}$，直接在稀疏矩阵 CSR 格式上进行运算。由于 $||\vec{x}_{old} - \vec{x}_{new}||_1$ 极小，迭代次数将从全量计算的近百次骤降到几次以内。该方法矩阵级操作高效，最易与 scipy.sparse 结合。

2. **局部推送算法 (Local Push Algorithm / Gauss-Southwell)**
   真正意义上的增量计算。在产生上述初始残差 $\vec{r}$ 后，维护一个优先队列或活跃节点集合，只对 $|r_i| > \epsilon$ 的节点进行操作：将残差累加到自身，并将新的残差按出度比例 Push 给下游邻居，直至全网残差消耗殆尽。

本项目的 Python 实现将主要构建并对比上述增量算法与传统重新计算（Restart）的性能差距。
```

### 下一步建议

如果你准备开始写 Python 代码实现增量算法，我建议你先实现 **Warm-Start (热启动) 法**作为 Baseline，因为它在 `scipy.sparse` 环境下非常好写（只需把初始向量从 `np.random.rand()` 改为传入旧的 `x`）。
然后再实现 **Local Push**，这需要你写一个 `while` 循环和队列来传播我在上面推导的那个 $\Delta \vec{r}$。需要我为你提供基于 Python 的 Local Push 算法骨架吗？