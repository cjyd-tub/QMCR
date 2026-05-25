import os
import sys
import math
import pprint

import torch

# 启用cuDNN优化，加速训练
torch.backends.cudnn.benchmark = True

# 启用TF32加速（RTX 30/40系列）
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import dataset, gnn, model, task, util


def train_and_validate(cfg, solver):
    if cfg.train.num_epoch == 0:
        return
    
    # 直接训练所有epoch
    kwargs = cfg.train.copy()
    solver.train(**kwargs)
    
    # 只在最后保存模型
    solver.save("model_final.pth")
    
    # 设置评估指标，包括新添加的ndcg指标
    solver.model.metric = ("mrr@20", "hits@20", "sd@20", "fhit@20", "lt@20", "ndcg@20", 
                          "mrr@10", "hits@10", "sd@10", "fhit@10", "lt@10", "ndcg@10")
    
    # 最后进行一次测试
    solver.evaluate("test")

    return solver


def test(cfg, solver):
    solver.model.metric = ( "hits@3", "hits@5", "hits@10", "hits@20", "hits@30","hits@40", "hits@50",  "mrr@3", "mrr@5", "mrr@10", "mrr@20", "mrr@30", "mrr@40", "mrr@50",
                            "fhit@3", "fhit@5", "fhit@10", "fhit@20", "fhit@30", "fhit@40", "fhit@50",  "sd@3", "sd@5", "sd@10", "sd@20", "sd@30", "sd@40", "sd@50",
                            "lt@3", "lt@5", "lt@10", "lt@20", "lt@30", "lt@40", "lt@50")
    solver.evaluate("test")


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))


    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)
    if args.type == 0:
        train_and_validate(cfg, solver)
    else:
        solver.load(args.checkpoint)
        test(cfg, solver)
