from collections import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint  # 梯度检查点

from torchdrug import core, data, utils
from torchdrug.core import Registry as R

from . import layer


@R.register("model.NBFNet")
class NeuralBellmanFordNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="distmult",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, dependent=True):
        super(NeuralBellmanFordNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        num_relation = int(num_relation)
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], num_relation,
                                                               self.dims[0], message_func,  layer_norm,
                                                               activation, dependent))

    def forward(self, graph, input, all_loss=None, metric=None):

        with graph.node():
            graph.boundary = input
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden
        node_query = graph.query.expand(graph.num_node, -1, -1)

        if self.concat_hidden:
            node_feature = torch.cat(hiddens + [node_query], dim=-1)
        else:
            node_feature = torch.cat([hiddens[-1], node_query], dim=-1)
        return {
            "node_feature": node_feature,
        }


@R.register("model.CompGCN")
class CompositionalGraphConvolutionalNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="mult", short_cut=False, layer_norm=False,
                 activation="relu", concat_hidden=False):
        super(CompositionalGraphConvolutionalNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        num_relation = int(num_relation)
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.CompositionalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                            message_func, layer_norm, activation))
        self.relation = nn.Embedding(num_relation, input_dim)

    def forward(self, graph, input, all_loss=None, metric=None):
        graph.relation_input = self.relation.weight
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        node_query = graph.query.expand(graph.num_node, -1, -1)
        if self.concat_hidden:
            node_feature = torch.cat(hiddens + [node_query], dim=-1)
        else:
            node_feature = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": node_feature,
        }


@R.register("model.GPRGNN")
class GeneralizedPageRankGNN(nn.Module, core.Configurable):
    """
    GPR-GNN: 基于广义PageRank的图神经网络
    核心思想：不只用最后一层，而是加权组合所有层的输出
    """

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="distmult",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, 
                 dependent=True, K=None, alpha=0.1, init_type="PPR"):
        super(GeneralizedPageRankGNN, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        num_relation = int(num_relation)
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        
        # GPR-GNN特有参数
        self.K = K if K is not None else len(self.dims) - 1  # 默认传播步数为层数
        self.alpha = alpha
        self.init_type = init_type
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], num_relation,
                                                               self.dims[0], message_func, layer_norm,
                                                               activation, dependent))
        # GPR-GNN核心：可学习的传播权重
        self.propagation_weights = nn.Parameter(torch.zeros(self.K + 1))
        
        # 预定义对齐层，避免动态创建
        self.align_layers = nn.ModuleList()
        for k in range(self.K + 1):
            if k == 0:
                # 第0层是输入层，可能需要对齐到最后一层的维度
                self.align_layers.append(nn.Linear(input_dim, hidden_dims[-1]))
            else:
                # 其他层已经是正确维度
                self.align_layers.append(nn.Identity())
        
        self._init_propagation_weights()

    def _init_propagation_weights(self):
        """初始化传播权重"""
        if self.init_type == "PPR":
            # PageRank风格初始化：近距离高权重
            for k in range(self.K + 1):
                self.propagation_weights.data[k] = self.alpha * (1 - self.alpha) ** k
        else:
            # 均匀初始化
            nn.init.uniform_(self.propagation_weights, 0, 1)
        
        # 归一化
        self.propagation_weights.data = F.softmax(self.propagation_weights.data, dim=0)

    def forward(self, graph, input, all_loss=None, metric=None):
        with graph.node():
            graph.boundary = input
            
        # 存储所有层的表示，从输入开始
        all_hiddens = [input]  # 第0层：原始输入
        layer_input = input

        # 逐层传播，收集每层的输出
        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            all_hiddens.append(hidden)  # 第k层：传播k步后的表示
            layer_input = hidden

        # GPR-GNN核心：加权组合所有层
        weights = F.softmax(self.propagation_weights, dim=0)
        
        # 确保只使用可用的层数
        num_available_layers = min(len(all_hiddens), self.K + 1)
        combined_hidden = torch.zeros_like(all_hiddens[-1])
        
        for k in range(num_available_layers):
            if k < len(weights):
                # 使用预定义的对齐层
                aligned_hidden = self.align_layers[k](all_hiddens[k])
                combined_hidden += weights[k] * aligned_hidden

        # 处理输出格式，与NBFNet保持一致
        node_query = graph.query.expand(graph.num_node, -1, -1)
        
        if self.concat_hidden:
            # 如果要拼接隐藏层，使用所有层
            node_feature = torch.cat(all_hiddens[1:] + [node_query], dim=-1)
        else:
            # 否则使用组合后的表示
            node_feature = torch.cat([combined_hidden, node_query], dim=-1)
        
        return {
            "node_feature": node_feature,
        }


@R.register("model.AdaptiveGPRGNN")
class AdaptiveGeneralizedPageRankGNN(nn.Module, core.Configurable):
    """
    Adaptive GPR-GNN: 自适应注意力版本的GPR-GNN
    核心改进：根据查询和节点特征，动态计算每层的权重
    不再使用固定的全局权重，而是根据输入自适应调整
    
    添加梯度检查点支持：降低显存占用，速度略慢
    """

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="distmult",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, 
                 dependent=True, K=None, alpha=0.1, init_type="PPR", attention_type="query_aware",
                 use_checkpoint=True):  # 新增：是否使用梯度检查点
        super(AdaptiveGeneralizedPageRankGNN, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        num_relation = int(num_relation)
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.attention_type = attention_type
        self.use_checkpoint = use_checkpoint  # 保存检查点设置
        
        # GPR-GNN特有参数
        self.K = K if K is not None else len(self.dims) - 1
        self.alpha = alpha
        self.init_type = init_type

        # 与NBFNet相同的层结构
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], num_relation,
                                                               self.dims[0], message_func, layer_norm,
                                                               activation, dependent))

        #  核心改进：自适应注意力模块（替代全局固定权重）
        self.attention_dim = hidden_dims[-1]
        
        # 预定义对齐层
        self.align_layers = nn.ModuleList()
        for k in range(self.K + 1):
            if k == 0:
                self.align_layers.append(nn.Linear(input_dim, hidden_dims[-1]))
            else:
                self.align_layers.append(nn.Identity())
        
        if self.attention_type == "query_aware":
            # Query-aware注意力：根据查询特征计算权重
            # 输入：查询特征 + 每层节点特征 → 输出：该层的注意力分数
            self.attention_query = nn.Sequential(
                nn.Linear(input_dim, self.attention_dim),
                nn.ReLU(),
                nn.Linear(self.attention_dim, self.attention_dim)
            )
            self.attention_key = nn.ModuleList([
                nn.Linear(self.attention_dim, self.attention_dim)
                for _ in range(self.K + 1)
            ])
            self.attention_scale = self.attention_dim ** 0.5
            
        elif self.attention_type == "mlp":
            # MLP注意力：直接对每层特征计算重要性分数
            self.attention_mlp = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.attention_dim + input_dim, self.attention_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.attention_dim // 2, 1)
                )
                for _ in range(self.K + 1)
            ])
        
        # 保留alpha作为初始化的先验
        self.prior_weights = nn.Parameter(torch.zeros(self.K + 1), requires_grad=False)
        self._init_prior_weights()
        
        if self.use_checkpoint:
            print("✅ AdaptiveGPRGNN: 梯度检查点已启用（降低显存，速度略慢）")

    def _init_prior_weights(self):
        """初始化先验权重（用作注意力的bias）"""
        if self.init_type == "PPR":
            for k in range(self.K + 1):
                self.prior_weights.data[k] = self.alpha * (1 - self.alpha) ** k
        else:
            self.prior_weights.data.fill_(1.0 / (self.K + 1))

    def compute_attention_weights(self, all_hiddens, query):
        """
        计算自适应注意力权重
        
        Args:
            all_hiddens: List[Tensor], 每层的节点表示 [num_nodes, batch_size, hidden_dim]
            query: Tensor, 查询特征 [num_nodes, batch_size, input_dim]
        
        Returns:
            weights: Tensor, 注意力权重 [K+1, num_nodes, batch_size, 1]
        """
        if self.attention_type == "query_aware":
            # Query-aware注意力机制
            # Step 1: 将查询映射到attention空间
            query_feature = self.attention_query(query)  # [num_nodes, batch_size, attention_dim]
            
            # Step 2: 计算每层与查询的相似度
            attention_scores = []
            for k in range(min(len(all_hiddens), self.K + 1)):
                # 对齐并投影该层特征
                aligned_hidden = self.align_layers[k](all_hiddens[k])  # [num_nodes, batch_size, attention_dim]
                key = self.attention_key[k](aligned_hidden)  # [num_nodes, batch_size, attention_dim]
                
                # 计算注意力分数（点积注意力）
                score = torch.sum(query_feature * key, dim=-1, keepdim=True) / self.attention_scale
                # [num_nodes, batch_size, 1]
                
                # 加入先验权重作为bias
                score = score + self.prior_weights[k]
                
                attention_scores.append(score)
            
            # Stack and softmax
            attention_scores = torch.stack(attention_scores, dim=0)  # [K+1, num_nodes, batch_size, 1]
            weights = F.softmax(attention_scores, dim=0)  # 在K维度上softmax
            
        elif self.attention_type == "mlp":
            # MLP注意力机制
            attention_scores = []
            for k in range(min(len(all_hiddens), self.K + 1)):
                aligned_hidden = self.align_layers[k](all_hiddens[k])
                # 拼接层特征和查询特征
                concat_feature = torch.cat([aligned_hidden, query], dim=-1)
                # 通过MLP计算分数
                score = self.attention_mlp[k](concat_feature)  # [num_nodes, batch_size, 1]
                score = score + self.prior_weights[k]
                attention_scores.append(score)
            
            attention_scores = torch.stack(attention_scores, dim=0)
            weights = F.softmax(attention_scores, dim=0)
        
        return weights

    def _layer_forward(self, gnn_layer, graph, layer_input):
        """单层GNN前向传播，用于梯度检查点"""
        return gnn_layer(graph, layer_input)

    def forward(self, graph, input, all_loss=None, metric=None):
        with graph.node():
            graph.boundary = input
            
        # 存储所有层的表示
        all_hiddens = [input]
        layer_input = input

        # 逐层传播（使用梯度检查点降低显存）
        for gnn_layer in self.layers:
            if self.use_checkpoint and self.training:
                # 训练时使用梯度检查点：不保存中间激活值，反向传播时重新计算
                # use_reentrant=False 是PyTorch推荐的新API
                hidden = checkpoint(
                    gnn_layer,
                    graph,
                    layer_input,
                    use_reentrant=False
                )
            else:
                # 推理时或不使用检查点时，正常前向传播
                hidden = gnn_layer(graph, layer_input)
            
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            all_hiddens.append(hidden)
            layer_input = hidden

        #  核心改进：自适应计算权重（替代固定权重）
        node_query = graph.query  # [num_nodes, batch_size, input_dim]
        
        # 计算自适应注意力权重
        weights = self.compute_attention_weights(all_hiddens, node_query)
        # weights shape: [K+1, num_nodes, batch_size, 1]
        
        # 加权组合所有层
        num_available_layers = min(len(all_hiddens), self.K + 1)
        combined_hidden = torch.zeros_like(all_hiddens[-1])
        
        for k in range(num_available_layers):
            aligned_hidden = self.align_layers[k](all_hiddens[k])
            # weights[k]: [num_nodes, batch_size, 1]
            # aligned_hidden: [num_nodes, batch_size, hidden_dim]
            combined_hidden += weights[k] * aligned_hidden

        # 处理输出格式
        node_query_expanded = node_query.expand(graph.num_node, -1, -1)
        
        if self.concat_hidden:
            node_feature = torch.cat(all_hiddens[1:] + [node_query_expanded], dim=-1)
        else:
            node_feature = torch.cat([combined_hidden, node_query_expanded], dim=-1)
        
        return {
            "node_feature": node_feature,
        }
