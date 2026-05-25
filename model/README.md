# 基于频谱域的图神经网络模型

本项目实现了多种图神经网络模型，包括基于空间域和频谱域的实现。

## 模型概览

### 空间域模型

1. **CompGCN**：组合图卷积网络，通过组合操作处理关系和节点特征
2. **NBFNet**：神经Bellman-Ford网络，基于关系图的消息传递神经网络

### 频谱域模型（新增）

1. **SpectralCompGCN**：基于频谱域的组合图卷积网络，使用图拉普拉斯矩阵特征分解
2. **AttSpectralCompGCN**：结合频谱域处理和多头注意力机制的增强版CompGCN
3. **SComGNN**：基于频谱补充关系图神经网络，分离低频和中频成分，使用两阶段注意力机制（最新）

## 频谱域图神经网络原理

频谱域图神经网络基于图信号处理理论，主要步骤：

1. **图拉普拉斯矩阵构建**：从图的邻接矩阵计算归一化拉普拉斯矩阵
2. **特征分解**：计算拉普拉斯矩阵的特征值和特征向量
3. **图傅立叶变换**：将节点特征投影到频谱域
4. **频谱域滤波**：应用可学习的滤波器
5. **逆图傅立叶变换**：将结果转回空间域

本实现使用Chebyshev多项式近似避免直接计算特征分解，提高计算效率。

## SComGNN模型亮点

SComGNN (Spectral-based Complementary Graph Neural Network) 是最新的频谱域GNN模型，基于论文《Spectral-Based Graph Neural Networks for Complementary Item Recommendation》，具有以下创新点：

1. **低频与中频成分分离**：同时捕获节点间的相关性（低频）和差异性（中频）
2. **特定频率滤波器**：使用低通滤波器捕获相关性，中通滤波器捕获差异性
3. **两阶段注意力机制**：
   - 第一阶段：特征级别注意力，自适应融合低频和中频特征
   - 第二阶段：节点级别多头自注意力，增强节点间信息交互
4. **高度可解释性**：频谱分离使得模型推荐结果具有明确的可解释性
5. **稀疏图优化**：针对边稀疏的图结构进行了特殊优化

## 频谱域与空间域模型对比

| 特性 | 空间域模型 | 频谱域模型 | SComGNN |
|------|-----------|-----------|---------|
| 计算复杂度 | 低 | 中等 | 中等 |
| 表达能力 | 局部结构 | 全局结构 | 全局结构+频率分离 |
| 参数效率 | 高 | 高 | 高 |
| 感受野 | 受层数限制 | 理论上覆盖全图 | 理论上覆盖全图 |
| 特征提取 | 局部特征聚合 | 频谱特征学习 | 多频段特征分离学习 |
| 可解释性 | 低 | 中 | 高 |
| 性能 | 基准 | 好 | 最佳 |

## 使用方法

### 模型配置

```python
# 使用标准CompGCN
model = CompositionalGraphConvolutionalNetwork(
    input_dim=128,
    hidden_dims=[256, 128],
    num_relation=10,
    message_func="mult",
    short_cut=True,
    layer_norm=True
)

# 使用频谱域CompGCN
spectral_model = SpectralCompositionalGraphConvolutionalNetwork(
    input_dim=128,
    hidden_dims=[256, 128],
    num_relation=10,
    chebyshev_k=3,  # Chebyshev多项式阶数
    short_cut=True,
    layer_norm=True
)

# 使用带注意力的频谱域CompGCN
att_spectral_model = AttentionSpectralCompGCN(
    input_dim=128,
    hidden_dims=[256, 128],
    num_relation=10,
    chebyshev_k=3,
    num_heads=4,  # 多头注意力头数
    short_cut=True,
    layer_norm=True,
    dropout=0.1
)

# 使用最新的SComGNN模型（推荐）
scom_model = SpectralComplementaryGNN(
    input_dim=128,
    hidden_dims=[256, 128],
    num_relation=10,
    chebyshev_k=5,  # 中频滤波器的Chebyshev多项式阶数
    num_heads=4,    # 多头注意力头数
    short_cut=True,
    layer_norm=True,
    dropout=0.1,
    concat_hidden=False
)
```

### SComGNN使用示例

我们提供了一个完整的示例脚本 `example.py`，演示如何使用SComGNN进行API推荐：

```python
# 运行SComGNN示例
python example.py
```

该示例将加载数据集，训练SComGNN模型，并评估其性能，最后与NBFNet等基线模型进行比较。

## 注意事项

1. 频谱域模型计算成本更高，但能捕获全局图结构信息
2. SComGNN中的低通滤波器和中通滤波器参数可以调整，以适应不同数据集的特性
3. `chebyshev_k`参数控制多项式阶数，增大可提高表达能力，但也增加计算量
4. 两阶段注意力机制对于稀疏图和稠密图都有良好的表现

## 性能提升要点

1. 频谱域处理能更好捕获全局结构关系
2. 频率分离使模型能够同时关注相关性和差异性
3. 两阶段注意力机制增强节点间的交互并平衡不同频率信息
4. 残差连接和层归一化提高训练稳定性
5. 针对稀疏图的特殊优化提高效率 