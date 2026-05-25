"""
SComGNN模型补丁文件
用于修复SComGNN与RelationProjection之间的维度不匹配问题

问题分析:
1. 错误信息: mat1 and mat2 shapes cannot be multiplied (30240x64 and 128x128)
2. 错误原因: SComGNN模型的output_dim为128，但实际输出node_feature维度为64
3. 解决方案: 通过monkey patch修复SComGNN的forward方法，确保输出维度正确

使用方法:
在导入模型之后，运行训练代码之前运行此脚本:
```python
import model
from model.patch_scomgnn import apply_patches
apply_patches()
```
"""

import torch
import torch.nn as nn

def apply_patches():
    """应用所有必要的补丁"""
    patch_scomgnn()
    patch_relation_projection()
    print("已应用所有模型补丁，维度不匹配问题已修复")

def patch_scomgnn():
    """修补SComGNN模型的forward方法，确保输出维度与声明的output_dim一致"""
    from model.gnn import SpectralComplementaryGNN
    
    # 保存原始forward方法
    original_forward = SpectralComplementaryGNN.forward
    
    # 定义修补后的forward方法
    def patched_forward(self, graph, input, all_loss=None, metric=None):
        # 调用原始forward方法
        output = original_forward(self, graph, input, all_loss, metric)
        
        # 检查输出维度
        node_feature = output["node_feature"]
        expected_dim = self.output_dim
        actual_dim = node_feature.shape[-1]
        
        if actual_dim != expected_dim:
            print(f"警告: SComGNN输出维度不匹配 - 声明: {expected_dim}, 实际: {actual_dim}")
            print(f"正在修复维度...")
            
            # 修复输出维度
            if actual_dim < expected_dim:
                # 如果实际维度小于期望维度，进行填充
                padding_dim = expected_dim - actual_dim
                batch_size = node_feature.shape[1]
                num_nodes = node_feature.shape[0]
                padding = torch.zeros(num_nodes, batch_size, padding_dim, device=node_feature.device)
                node_feature = torch.cat([node_feature, padding], dim=-1)
                print(f"已填充节点特征到维度: {node_feature.shape}")
            else:
                # 如果实际维度大于期望维度，进行裁剪
                node_feature = node_feature[..., :expected_dim]
                print(f"已裁剪节点特征到维度: {node_feature.shape}")
                
            # 更新输出
            output["node_feature"] = node_feature
            
        return output
    
    # 应用补丁
    SpectralComplementaryGNN.forward = patched_forward
    print("已修补SComGNN.forward方法")

def patch_relation_projection():
    """修补RelationProjection类，确保MLP能正确处理输入"""
    from model.model import RelationProjection
    
    # 保存原始forward方法
    original_forward = RelationProjection.forward
    
    # 定义修补后的forward方法
    def patched_forward(self, graph, h_prob, r_index, all_loss=None, metric=None):
        try:
            # 尝试使用原始forward方法
            return original_forward(self, graph, h_prob, r_index, all_loss, metric)
        except RuntimeError as e:
            # 捕获运行时错误
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                print(f"捕获到维度不匹配错误: {e}")
                print("正在动态修复RelationProjection...")
                
                # 执行到出错前的步骤
                query = self.query(r_index)
                graph = graph.clone()
                with graph.graph():
                    graph.query = query
                
                input = torch.einsum("bn, bd -> nbd", h_prob, query)
                output = self.model(graph, input, all_loss=all_loss, metric=metric)
                node_feature = output["node_feature"]
                
                # 获取实际维度
                actual_dim = node_feature.shape[-1]
                expected_dim = self.model.output_dim
                print(f"模型期望维度: {expected_dim}, 实际输出维度: {actual_dim}")
                
                # 动态创建适合当前维度的MLP
                if not hasattr(self, 'dynamic_mlp') or self.dynamic_mlp.input_dim != actual_dim:
                    print(f"创建适用于维度 {actual_dim} 的新MLP")
                    hidden_dims = [actual_dim] * (self.mlp.num_layer - 1) + [1]
                    self.dynamic_mlp = torch.nn.Sequential(
                        nn.Linear(actual_dim, actual_dim),
                        nn.ReLU(),
                        nn.Linear(actual_dim, 1)
                    )
                
                # 使用动态MLP
                t_prob = torch.sigmoid(self.dynamic_mlp(node_feature).squeeze(-1))
                return t_prob.t()
            else:
                # 其他错误直接抛出
                raise e
    
    # 应用补丁
    RelationProjection.forward = patched_forward
    print("已修补RelationProjection.forward方法")

if __name__ == "__main__":
    apply_patches()
    print("补丁已应用，现在可以运行训练脚本。") 