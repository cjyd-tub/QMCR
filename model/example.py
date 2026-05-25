import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchdrug import data, models, tasks, metrics
from torchdrug.core import Registry as R
from torchdrug.utils import comm

from .gnn import CompositionalGraphConvolutionalNetwork, NeuralBellmanFordNetwork
from .gnn import SpectralCompositionalGraphConvolutionalNetwork, AttentionSpectralCompGCN
from .model import QueryExecutor
from .task import RelationPrediction
from .data import LogicalQuery, load_dataset

def run_experiment(model_type, dataset_name="PWA", hidden_dims=[256, 128], num_epoch=20):
    """
    运行指定模型类型的实验
    
    参数:
        model_type: 模型类型，可以是'CompGCN', 'NBFNet', 'SpectralCompGCN', 'AttSpectralCompGCN'之一
        dataset_name: 数据集名称，'PWA'或'HGA'
        hidden_dims: 隐藏层维度
        num_epoch: 训练轮数
    
    返回:
        测试结果指标
    """
    # 加载数据集
    train_set, valid_set, test_set = load_dataset(dataset_name)
    
    # 模型参数
    num_relation = train_set.num_relation
    input_dim = 128
    
    # 根据model_type选择不同的模型
    if model_type == "CompGCN":
        gnn = CompositionalGraphConvolutionalNetwork(
            input_dim=input_dim, 
            hidden_dims=hidden_dims, 
            num_relation=num_relation, 
            message_func="mult", 
            short_cut=True, 
            layer_norm=True
        )
    elif model_type == "NBFNet":
        gnn = NeuralBellmanFordNetwork(
            input_dim=input_dim, 
            hidden_dims=hidden_dims, 
            num_relation=num_relation, 
            message_func="distmult", 
            short_cut=True, 
            layer_norm=True
        )
    elif model_type == "SpectralCompGCN":
        gnn = SpectralCompositionalGraphConvolutionalNetwork(
            input_dim=input_dim, 
            hidden_dims=hidden_dims, 
            num_relation=num_relation, 
            chebyshev_k=3, 
            short_cut=True, 
            layer_norm=True
        )
    elif model_type == "AttSpectralCompGCN":
        gnn = AttentionSpectralCompGCN(
            input_dim=input_dim, 
            hidden_dims=hidden_dims, 
            num_relation=num_relation, 
            chebyshev_k=3,
            num_heads=4,
            short_cut=True, 
            layer_norm=True,
            dropout=0.1
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    # 构建查询执行器
    model = QueryExecutor(gnn, dropout_ratio=0.1)
    
    # 设置任务
    task = RelationPrediction(model, train_set.num_entity, train_set.num_relation)
    
    # 优化器
    optimizer = optim.Adam(task.parameters(), lr=1e-3)
    
    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=128, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=128, num_workers=4)
    
    # 训练循环
    best_valid_score = 0
    best_test_score = None
    
    for epoch in range(num_epoch):
        # 训练阶段
        model.train()
        for batch in train_loader:
            batch = comm.cuda(batch)
            loss, metric = task(batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            # 验证集评估
            valid_metric = {}
            for batch in valid_loader:
                batch = comm.cuda(batch)
                _, metric = task(batch)
                for k, v in metric.items():
                    valid_metric[k] = valid_metric.get(k, 0) + v.item() * len(batch)
            for k in valid_metric:
                valid_metric[k] /= len(valid_set)
            
            valid_score = valid_metric["mr"]  # 平均排名，越小越好
            
            # 记录最佳模型
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                
                # 在测试集上评估
                test_metric = {}
                for batch in test_loader:
                    batch = comm.cuda(batch)
                    _, metric = task(batch)
                    for k, v in metric.items():
                        test_metric[k] = test_metric.get(k, 0) + v.item() * len(batch)
                for k in test_metric:
                    test_metric[k] /= len(test_set)
                    
                best_test_score = test_metric
            
        print(f"Epoch {epoch}, Valid: {valid_metric}, Test: {test_metric if 'test_metric' in locals() else 'N/A'}")
    
    return best_test_score

def compare_models():
    """比较不同模型的性能"""
    models = ["CompGCN", "NBFNet", "SpectralCompGCN", "AttSpectralCompGCN"]
    results = {}
    
    for model_type in models:
        print(f"正在评估 {model_type}...")
        result = run_experiment(model_type)
        results[model_type] = result
    
    # 打印比较结果
    print("\n模型性能比较:")
    print("=" * 80)
    print(f"{'模型类型':<20} {'MRR':<10} {'Hits@1':<10} {'Hits@3':<10} {'Hits@10':<10}")
    print("-" * 80)
    for model_type, result in results.items():
        print(f"{model_type:<20} {result['mrr']:<10.4f} {result['hits@1']:<10.4f} {result['hits@3']:<10.4f} {result['hits@10']:<10.4f}")
    print("=" * 80)

def train_api_recommendation_with_scomgnn():
    """
    使用基于频谱补充关系图神经网络(SComGNN)的API推荐示例
    
    该示例展示了如何使用SComGNN进行API关系推荐，并与NBFNet等基线模型进行比较
    """
    import torch
    import argparse
    import numpy as np
    from torchdrug import core, datasets, models, tasks, utils
    from torchdrug.utils import comm
    
    # 从内部模块导入所需组件
    from model import gnn, task, layer, dataset
    
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    
    # 设置数据集参数
    args.dataset = "PWA"  # 可选: "PWA", "HGA"
    args.model = "SComGNN"  # 我们的新模型
    args.task = "LogicalQuery"
    args.hidden_dims = [64, 64]
    
    # 训练参数
    args.batch_size = 32
    args.num_epoch = 10
    args.learning_rate = 1e-3
    args.weight_decay = 0
    
    # SComGNN特有参数
    args.chebyshev_k = 5
    args.num_heads = 4
    args.layer_norm = True
    args.dropout = 0.1
    
    # 创建数据集
    print("加载数据集...")
    dataset = getattr(dataset, "APIRelationDataset")(args.dataset)
    num_entity = dataset.num_entity
    num_relation = dataset.num_relation
    
    # 数据集分割
    lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
    lengths += [len(dataset) - sum(lengths)]
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)
    print("数据集大小: 训练集 %d, 验证集 %d, 测试集 %d" % tuple(lengths))
    
    # 创建SComGNN模型
    print("构建SComGNN模型...")
    model = getattr(gnn, args.model)(
        input_dim=32,
        hidden_dims=args.hidden_dims,
        num_relation=num_relation,
        chebyshev_k=args.chebyshev_k,
        num_heads=args.num_heads,
        layer_norm=args.layer_norm,
        dropout=args.dropout,
        short_cut=True,
        concat_hidden=False
    )
    
    # 创建逻辑查询任务
    logical_task = getattr(task, args.task)(
        model=gnn.QueryExecutor(model),
        dataset=args.dataset,
        criterion="bce",
        metric=("mrr@3", "hits@3", "sd@3", "fhit@3", "lt@3")
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        logical_task.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.6, patience=5, verbose=True
    )
    
    # 将模型移动到GPU（如果可用）
    if torch.cuda.is_available():
        logical_task = logical_task.cuda()
    
    # 创建训练引擎
    engine = core.Engine(
        logical_task,
        train_set,
        valid_set,
        test_set,
        optimizer,
        scheduler=scheduler,
        batch_size=args.batch_size,
        gpus=[0] if torch.cuda.is_available() else None
    )
    
    # 训练模型
    print("开始训练...")
    # 定义回调函数
    def create_structure():
        return {"mrr": [], "hits": [], "sd": [], "fhit": [], "lt": []}
    
    # 存储每个epoch的指标
    metrics = {
        "train": create_structure(),
        "valid": create_structure(),
        "test": create_structure()
    }
    
    # 收集训练过程中的指标
    for epoch in range(args.num_epoch):
        train_loss = []
        engine.train()
        for batch in engine.train_loader:
            loss, train_metric = engine.train_step(batch)
            train_loss.append(loss.item())
        
        # 验证
        engine.evaluate()
        
        # 收集指标
        for k, v in engine.meter.items():
            if k.endswith("mrr@3"):
                metrics["train" if "train" in k else "valid" if "valid" in k else "test"]["mrr"].append(v)
            elif k.endswith("hits@3"):
                metrics["train" if "train" in k else "valid" if "valid" in k else "test"]["hits"].append(v)
            elif k.endswith("sd@3"):
                metrics["train" if "train" in k else "valid" if "valid" in k else "test"]["sd"].append(v)
            elif k.endswith("fhit@3"):
                metrics["train" if "train" in k else "valid" if "valid" in k else "test"]["fhit"].append(v)
            elif k.endswith("lt@3"):
                metrics["train" if "train" in k else "valid" if "valid" in k else "test"]["lt"].append(v)
        
        # 打印当前进度
        print("Epoch %d/%d: 训练损失 %.4f, 验证 MRR@3 %.4f, 验证 Hits@3 %.4f" %
              (epoch + 1, args.num_epoch, np.mean(train_loss), 
               metrics["valid"]["mrr"][-1], metrics["valid"]["hits"][-1]))
    
    # 测试阶段
    print("测试中...")
    engine.evaluate("test")
    for k, v in engine.meter.items():
        if "test" in k:
            print("%s: %.4f" % (k, v))
    
    # 比较与NBFNet的性能
    print("\n与基线模型比较:")
    print("%-10s %-10s %-10s %-10s" % ("模型", "MRR@3", "Hits@3", "功能匹配"))
    print("%-10s %-10.4f %-10.4f %-10.4f" % (
        "SComGNN", metrics["test"]["mrr"][-1], metrics["test"]["hits"][-1], metrics["test"]["fhit"][-1]
    ))
    print("%-10s %-10.4f %-10.4f %-10.4f" % (
        "NBFNet", 0.267, 0.412, 0.385
    ))  # NBFNet的基线数据（示例值）

if __name__ == "__main__":
    train_api_recommendation_with_scomgnn() 