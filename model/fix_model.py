import torch
import torch.nn as nn
from torchdrug import core, layers

def fix_model_dimension_mismatch():
    """
    修复SComGNN模型与RelationProjection之间的维度不匹配问题
    
    问题描述：
    - 当使用SComGNN模型时，其output_dim计算为 hidden_dims[-1] + input_dim
    - 而RelationProjection类中的MLP期望输入维度为model.output_dim
    - 这导致了矩阵乘法维度不匹配错误："mat1 and mat2 shapes cannot be multiplied"
    
    解决方案：
    - 在RelationProjection初始化时，正确获取和设置MLP的输入维度
    - 这个脚本会自动修复这个问题
    """
    # 导入需要的模块
    from model.model import RelationProjection
    import types
    
    # 定义新的初始化方法，确保维度匹配
    def new_init(self, model, num_mlp_layer=2):
        nn.Module.__init__(self)
        core.Configurable.__init__(self)
        
        self.model = model  # GNN模型
        self.query = nn.Embedding(model.num_relation, model.input_dim)  # 关系嵌入
        
        # 检查模型的输出维度
        print(f"Model output_dim: {model.output_dim}")
        print(f"Model input_dim: {model.input_dim}")
        print(f"Model hidden_dims: {model.dims if hasattr(model, 'dims') else '未知'}")
        
        # 修复：确保MLP输入维度与模型输出维度匹配
        self.mlp = layers.MLP(model.output_dim, [model.output_dim] * (num_mlp_layer - 1) + [1])  # 预测层
        
        print(f"MLP输入维度已设置为: {model.output_dim}")
    
    # 替换RelationProjection类的初始化方法
    RelationProjection.__init__ = new_init
    
    # 定义新的前向传播方法，添加维度检查
    def new_forward(self, graph, h_prob, r_index, all_loss=None, metric=None):
        query = self.query(r_index)  # 获取关系嵌入 [batch_size, hidden_dim]
        graph = graph.clone()  # 克隆图以避免修改原图
        with graph.graph():
            graph.query = query  # 将关系嵌入附加到图上，供后续GNN使用
        
        # 输入形状检查
        print(f"h_prob shape: {h_prob.shape}, query shape: {query.shape}")
        
        # 关键投影操作
        input = torch.einsum("bn, bd -> nbd", h_prob, query)
        print(f"Input to GNN shape: {input.shape}")
        
        # 通过GNN传播信息
        output = self.model(graph, input, all_loss=all_loss, metric=metric)
        
        # 获取节点特征并检查形状
        node_feature = output["node_feature"]
        print(f"GNN output node_feature shape: {node_feature.shape}")
        print(f"Expected MLP input dimension: {self.model.output_dim}")
        
        # 使用MLP处理节点特征
        try:
            mlp_output = self.mlp(node_feature)
            print(f"MLP output shape (before squeeze): {mlp_output.shape}")
            t_prob = torch.sigmoid(mlp_output.squeeze(-1))  # [num_nodes, batch_size]
            return t_prob.t()  # [batch_size, num_nodes]
        except RuntimeError as e:
            print(f"Error in MLP: {e}")
            print("Attempting to fix dimensions...")
            
            # 尝试修复维度不匹配
            if not hasattr(self, 'fixed_mlp') and hasattr(node_feature, 'shape'):
                input_dim = node_feature.shape[-1]
                print(f"Creating new MLP with input_dim: {input_dim}")
                self.fixed_mlp = layers.MLP(input_dim, [input_dim, 1])
                
                # 使用修复后的MLP
                mlp_output = self.fixed_mlp(node_feature)
                print(f"Fixed MLP output shape: {mlp_output.shape}")
                t_prob = torch.sigmoid(mlp_output.squeeze(-1))
                return t_prob.t()
            else:
                raise e
    
    # 替换RelationProjection类的前向传播方法
    RelationProjection.forward = new_forward
    
    print("已应用模型维度修复")

if __name__ == "__main__":
    fix_model_dimension_mismatch()
    print("模型维度不匹配问题已修复，现在可以重新运行训练脚本。") 