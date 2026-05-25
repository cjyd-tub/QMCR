import os
import pickle
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from .data import Query


class LogicalQueryDataset(data.KnowledgeGraphDataset):
    """Logical query dataset."""

    struct2type = {
        # 基础查询模式
        ("e", ("r",)): "base-P",
        
        # 交集查询
        (("e", ("r",)), ("e", ("r",))): "2I",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r",))): "3I",

        # 并集查询
        (("e", ("r",)), ("e", ("r",)), ("u",)): "2U",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r",)), ("u",)): "3U",
        
        # 复杂查询结构
        ((("e", ("r",)), ("e", ("r",))), (("e", ("r",)), ("e", ("r",))), ("u",)): "2I-2I-U",
        ((("e", ("r",)), ("e", ("r",)), ("e", ("r",))), (("e", ("r",)), ("e", ("r",)), ("e", ("r",))), ("u",)): "3I-3I-U",
        
        # 其他复杂查询结构
        ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))): "2U-1N-I",
        (((("e", ("r",)),("e", ("r",))), (("e", ("r",)),("e", ("r",))), ("u",)), ((("e", ("r",)), ("e", ("r",)),("u",)),("n",))): "2I-2I-U-2UN-I",
        (((("e", ("r",)),("e", ("r",)),("e", ("r",))), (("e", ("r",)),("e", ("r",)),("e", ("r",))), ("u",)), ((("e", ("r",)), ("e", ("r",)), ("e", ("r",)),("u",)),("n",))): "3I-3I-U-3UN-I",
        
        # 强交非替补并弱互补
        ((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)): "SC-NS-WC-U",

        # 二阶查询：两个一阶查询的交集
        (((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)),
         ((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",))): "SC-NS-WC-U-I-SC-NS-WC-U",

        # 三阶查询：三个一阶查询的交集
        (((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)),
         ((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)),
         ((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",))): "SC-NS-WC-U-I-SC-NS-WC-U-I-SC-NS-WC-U",
         
        # 一阶查询：(((API1强互补交API1替补取反)并(API1弱互补))并API1属性互补)
        (((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)): "SC-NS-WC-U-ATTR-U",

        # 二阶查询：一阶查询API1 交 一阶查询API2
        (
            ((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",),
            ((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)
        ): "SC-NS-WC-U-ATTR-U-I-SC-NS-WC-U-ATTR-U",

        # 三阶查询：一阶查询API1 交 一阶查询API2 交 一阶查询API3
        (
            ((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",),
            ((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",),
            ((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)
        ): "SC-NS-WC-U-ATTR-U-I-SC-NS-WC-U-ATTR-U-I-SC-NS-WC-U-ATTR-U",

        # 额外添加可能的格式变体
        ((("e", ("r",)), ("e", ("r", "n"))), ("e", ("r",)), ("u",), ((("e", ("r",)), ("e", "n")), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)): "SC-NS-WC-U-ATTR-U-I-SC-NS-WC-U-ATTR-U",
        
        (((("e", ("r",)), ("e", "n")), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)): "SC-NS-WC-U-ATTR-U-VAR",
        
        ((((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)),
         (((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)),
         (((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",))): "SC-NS-WC-U-ATTR-U-I-SC-NS-WC-U-ATTR-U-I-SC-NS-WC-U-ATTR-U-ALT",
        
        # 消融弱互补的查询结构（Ablation of Weak Complementary）
        # 一阶查询：(API1强互补交API1替补取反)
        (("e", ("r",)), ("e", ("r", 'n'))): "SC-NS",
        
        # 一阶查询：((API1强互补交API1替补取反)并API1属性互补)
        # 注意：此结构也用于消融强互补（Ablation of Strong Complementary）
        # 消融CC时查询中使用关系1（弱互补）代替关系0（强互补）
        ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)): "SC-NS-ATTR-U",
        
        # 二阶查询：(API1强互补交API1替补取反) 交 (API2强互补交API2替补取反)
        ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n')))): "SC-NS-I-SC-NS",
        
        # 二阶查询：((API1强互补交API1替补取反)并API1属性互补) 交 ((API2强互补交API2替补取反)并API2属性互补)
        # 注意：此结构也用于消融CC（去除强互补）
        (
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)),
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",))
        ): "SC-NS-ATTR-U-I-SC-NS-ATTR-U",
        
        # 三阶查询：三个(API强互补交API替补取反)的交集
        ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n')))): "SC-NS-I-SC-NS-I-SC-NS",
        
        # 三阶查询：三个((API强互补交API替补取反)并API属性互补)的交集
        # 注意：此结构也用于消融CC（去除强互补）
        (
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)),
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)),
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",))
        ): "SC-NS-ATTR-U-I-SC-NS-ATTR-U-I-SC-NS-ATTR-U",

        # 消融弱互补的查询结构 - 精确匹配生成代码的格式
        # 一阶查询：((API强互补交API替补取反)并API属性互补)
        ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)): "SC-NS-ATTR-U-ABLATION-WC",

        # 二阶查询：((API1强互补交API1替补取反)并API1属性互补) 交 ((API2强互补交API2替补取反)并API2属性互补)
        (
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)),
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",))
        ): "SC-NS-ATTR-U-I-SC-NS-ATTR-U-ABLATION-WC",

        # 三阶查询：三个((API强互补交API替补取反)并API属性互补)的交集
        (
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)),
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)),
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",))
        ): "SC-NS-ATTR-U-I-SC-NS-ATTR-U-I-SC-NS-ATTR-U-ABLATION-WC",
        
        # 消融弱互补的简化查询结构（不包含属性互补的情况）
        # 一阶：(API强互补交API替补取反)
        (("e", ("r",)), ("e", ("r", 'n'))): "SC-NS-ABLATION-WC",
        
        # 二阶：(API1强互补交API1替补取反) 交 (API2强互补交API2替补取反)
        ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n')))): "SC-NS-I-SC-NS-ABLATION-WC",
        
        # 三阶：三个(API强互补交API替补取反)的交集
        ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n')))): "SC-NS-I-SC-NS-I-SC-NS-ABLATION-WC",
        
        # 消融替补取反的查询结构 - 精确匹配生成代码的格式
        # 一阶查询：((API强互补交API替补取反)并API属性互补) - PWA数据集用
        ((("e", ("r",)), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)): "SC-WC-U-ATTR-U-ABLATION-SUB",
        
        # 二阶查询：两个完整一阶查询的交集
        (
            ((("e", ("r",)), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)),
            ((("e", ("r",)), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",))
        ): "SC-WC-U-ATTR-U-I-SC-WC-U-ATTR-U-ABLATION-SUB",
        
        # 三阶查询：三个完整一阶查询的交集
        (
            ((("e", ("r",)), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)),
            ((("e", ("r",)), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)),
            ((("e", ("r",)), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",))
        ): "SC-WC-U-ATTR-U-I-SC-WC-U-ATTR-U-I-SC-WC-U-ATTR-U-ABLATION-SUB",
        
        # 消融替补取反的简化查询结构（非PWA数据集，如HGA）
        # 一阶：(强互补并弱互补) - 注意：这与2U结构相同，但语义不同，代表消融替补取反
        # (("e", ("r",)), ("e", ("r",)), ("u",)): "SC-WC-U-ABLATION-SUB",  # 暂时注释，因为与2U冲突
        
        # 二阶：(API1强互补并API1弱互补) 交 (API2强互补并API2弱互补)
        ((("e", ("r",)), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",))): "SC-WC-U-I-SC-WC-U-ABLATION-SUB",
        
        # 三阶：三个(API强互补并API弱互补)的交集
        ((("e", ("r",)), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",))): "SC-WC-U-I-SC-WC-U-I-SC-WC-U-ABLATION-SUB",
        
        # ============ 查询结构2.0：弱互补也与替补取反交操作 ============
        # 一阶基础：(强互补交替补取反)并(弱互补交替补取反) 
        ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)): "SC-NS-I-WC-NS-U-V2",
        
        # 一阶完整：((强互补交替补取反)并(弱互补交替补取反))并属性互补
        (((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)): "SC-NS-I-WC-NS-U-ATTR-U-V2",
        
        # 二阶完整：两个一阶完整查询的交集
        (
            (((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)),
            (((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",))
        ): "SC-NS-I-WC-NS-U-ATTR-U-I-SC-NS-I-WC-NS-U-ATTR-U-V2",
        
        # 三阶完整：三个一阶完整查询的交集
        (
            (((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)),
            (((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)),
            (((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",))
        ): "SC-NS-I-WC-NS-U-ATTR-U-I-SC-NS-I-WC-NS-U-ATTR-U-I-SC-NS-I-WC-NS-U-ATTR-U-V2",

        # HGA查询结构2.0 - 不含属性互补版本
        # 二阶：两个一阶基础查询的交集
        (
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)),
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",))
        ): "SC-NS-I-WC-NS-U-I-SC-NS-I-WC-NS-U-V2",

        # 三阶：三个一阶基础查询的交集
        (
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)),
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)),
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",))
        ): "SC-NS-I-WC-NS-U-I-SC-NS-I-WC-NS-U-I-SC-NS-I-WC-NS-U-V2",

        # ============ 查询结构3.0：属性互补也与替补取反交操作（笔记严格版本） ============
        # 一阶完整：(((强互补∩¬替补)∪(弱互补∩¬替补))∪((供应商∪语义)∩¬替补))
        # 注意：这里属性互补是先并集再与替补取反，而不是分别与替补取反再并集
        (
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)),  # (SC∩¬SUB) ∪ (WC∩¬SUB)
            ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))),  # (Vendor ∪ Semantic) ∩ ¬SUB
            ("u",)
        ): "SC-NS-I-WC-NS-U-ATTR-U-NS-V3",
        
        # 二阶完整：两个一阶完整查询的交集
        (
            (((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",)),
            (((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",))
        ): "SC-NS-I-WC-NS-U-ATTR-U-NS-I-SC-NS-I-WC-NS-U-ATTR-U-NS-V3",
        
        # 三阶完整：三个一阶完整查询的交集
        (
            (((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",)),
            (((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",)),
            (((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",))
        ): "SC-NS-I-WC-NS-U-ATTR-U-NS-I-SC-NS-I-WC-NS-U-ATTR-U-NS-I-SC-NS-I-WC-NS-U-ATTR-U-NS-V3",

        # ============ 查询结构修改版V4：((SC∪WC)∩¬SUB) ∪ (ATTR∩¬SUB) ============
        # 与V3的区别：V3是((SC∩¬SUB)∪(WC∩¬SUB))∪(ATTR∩¬SUB)，强弱互补分别去噪
        # V4是((SC∪WC)∩¬SUB)∪(ATTR∩¬SUB)，强弱互补先合并再统一去噪
        # 一阶查询：((强互补∪弱互补)∩¬替补) ∪ ((供应商∪语义)∩¬替补)
        (
            ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))),  # (SC∪WC)∩¬SUB
            ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))),  # (Vendor∪Semantic)∩¬SUB
            ("u",)
        ): "SC-WC-U-NS-I-ATTR-U-NS-V4",
        
        # 二阶查询：Q1(a1) ∩ Q1(a2)
        (
            (((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",)),
            (((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",))
        ): "SC-WC-U-NS-I-ATTR-U-NS-I-SC-WC-U-NS-I-ATTR-U-NS-V4",
        
        # 三阶查询：Q1(a1) ∩ Q1(a2) ∩ Q1(a3)
        (
            (((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",)),
            (((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",)),
            (((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",))
        ): "SC-WC-U-NS-I-ATTR-U-NS-I-SC-WC-U-NS-I-ATTR-U-NS-I-SC-WC-U-NS-I-ATTR-U-NS-V4",

        # ============ 查询结构修改版V20：(SC∩¬SUB) ∪ WC ∪ (ATTR∩¬SUB) ============
        # 核心创新：弱互补不进行去噪，直接使用；强互补和属性互补独立去噪
        # 理论依据：弱互补是功能级共现，不存在API级替代噪声
        # 一阶查询：(强互补∩替补¬) ∪ 弱互补 ∪ (属性互补∩替补¬)
        (
            ((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)),  # (SC∩¬SUB) ∪ WC
            ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))),  # (Vendor∪Semantic)∩¬SUB
            ("u",)
        ): "SC-NS-U-WC-U-ATTR-NS-V20",
        
        # 二阶查询：Q1(a1) ∩ Q1(a2)
        (
            (((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",)),
            (((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",))
        ): "SC-NS-U-WC-U-ATTR-NS-I-SC-NS-U-WC-U-ATTR-NS-V20",
        
        # 三阶查询：Q1(a1) ∩ Q1(a2) ∩ Q1(a3)
        (
            (((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",)),
            (((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",)),
            (((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",)), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",))
        ): "SC-NS-U-WC-U-ATTR-NS-I-SC-NS-U-WC-U-ATTR-NS-I-SC-NS-U-WC-U-ATTR-NS-V20",

        # ============ 查询结构修改版V21：(SC∩¬SUB) ∪ ((WC∪ATTR)∩¬SUB) ============
        # 核心创新：强互补独立去噪；弱互补和属性互补合并后统一去噪
        # 理论依据：探索不同去噪策略对推荐效果的影响
        
        # V21中间结构1：弱互补 ∪ 属性互补
        (("e", ("r",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)): "WC-U-ATTR-V21",
        
        # V21中间结构2：(弱互补 ∪ 属性互补) ∩ 替补¬
        ((("e", ("r",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ("e", ("r", 'n'))): "WC-ATTR-U-NS-V21",
        
        # 一阶查询：(强互补∩替补¬) ∪ ((弱互补∪属性互补)∩替补¬)
        (
            (("e", ("r",)), ("e", ("r", 'n'))),  # SC∩¬SUB
            ((("e", ("r",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ("e", ("r", 'n'))),  # (WC∪ATTR)∩¬SUB
            ("u",)
        ): "SC-NS-U-WC-ATTR-U-NS-V21",
        
        # 二阶查询：Q1(a1) ∩ Q1(a2)
        (
            ((("e", ("r",)), ("e", ("r", 'n'))), ((("e", ("r",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ("e", ("r", 'n'))), ("u",)),
            ((("e", ("r",)), ("e", ("r", 'n'))), ((("e", ("r",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ("e", ("r", 'n'))), ("u",))
        ): "SC-NS-U-WC-ATTR-U-NS-I-SC-NS-U-WC-ATTR-U-NS-V21",
        
        # 三阶查询：Q1(a1) ∩ Q1(a2) ∩ Q1(a3)
        (
            ((("e", ("r",)), ("e", ("r", 'n'))), ((("e", ("r",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ("e", ("r", 'n'))), ("u",)),
            ((("e", ("r",)), ("e", ("r", 'n'))), ((("e", ("r",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ("e", ("r", 'n'))), ("u",)),
            ((("e", ("r",)), ("e", ("r", 'n'))), ((("e", ("r",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ("e", ("r", 'n'))), ("u",))
        ): "SC-NS-U-WC-ATTR-U-NS-I-SC-NS-U-WC-ATTR-U-NS-I-SC-NS-U-WC-ATTR-U-NS-V21",

        # ============ 消融弱互补查询结构映射 ============
        # 根据错误日志添加消融脚本生成的具体查询结构
        
        # 一阶查询：(强互补∩替补¬) ∪ (属性互补∩替补¬) - 消融弱互补版本
        ((("e", ("r",)), ("e", ("r", 'n'))), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",)): "SC-NS-U-ATTR-NS-ABLATION-WC",
        
        # 二阶查询：两个消融弱互补一阶查询的交集
        (((("e", ("r",)), ("e", ("r", 'n'))), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",)), ((("e", ("r",)), ("e", ("r", 'n'))), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",))): "SC-NS-U-ATTR-NS-I-SC-NS-U-ATTR-NS-ABLATION-WC",
        
        # 三阶查询：三个消融弱互补一阶查询的交集  
        (((("e", ("r",)), ("e", ("r", 'n'))), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",)), ((("e", ("r",)), ("e", ("r", 'n'))), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",)), ((("e", ("r",)), ("e", ("r", 'n'))), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",))): "SC-NS-U-ATTR-NS-I-SC-NS-U-ATTR-NS-I-SC-NS-U-ATTR-NS-ABLATION-WC",

        # ============ 消融语义互补/供应商互补的中间结构 ============
        # (WC∪单属性关系)∩¬SUB 中间结构
        # 用于 base_add_xiaoyuyi.py（消融语义互补）和 base_add_xiaogongying.py（消融供应商互补）
        ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))): "WC-ATTR-U-NS-ABLATION-SINGLE",
        
        # 处理更复杂的嵌套结构 - 修复后的正确语法格式
        # 错误结构1：修复关系部分的嵌套问题
        (((("e", ("r",)), ("e", ("r", 'n'))), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",))): "COMPLEX-NESTED-ABLATION-WC-1-FIXED",
        
        # 如果还有其他复杂结构，可以继续添加
        # 这个结构基于消融脚本可能生成的嵌套模式
        (((("e", ("r",)), ("e", ("r", 'n'))), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ((("e", ("r",)), ("e", ("r", 'n'))), ((("e", ("r",)), ("e", ("r",)), ("u",)), ("e", ("r", 'n'))), ("u",))): "COMPLEX-NESTED-ABLATION-WC-2-FIXED",

        # ============ 消融实验具体查询结构（基于代码实际逻辑） ============
        
        # 1. 消融属性互补的查询结构
        # 消融后结构: (强互补∩替补¬) ∪ (弱互补∩替补¬)
        
        # 一阶：(强互补∩替补¬) ∪ (弱互补∩替补¬)
        (
            (("e", ("r",)), ("e", ("r", 'n'))),  # SC∩¬SUB
            (("e", ("r",)), ("e", ("r", 'n'))),  # WC∩¬SUB
            ("u",)
        ): "SC-NS-U-WC-NS-ABLATION-ATTR",
        
        # 二阶：两个消融属性互补一阶查询的交集
        (
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)),
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",))
        ): "SC-NS-U-WC-NS-I-SC-NS-U-WC-NS-ABLATION-ATTR",
        
        # 三阶：三个消融属性互补一阶查询的交集
        (
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)),
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",)),
            ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), ("u",))
        ): "SC-NS-U-WC-NS-I-SC-NS-U-WC-NS-I-SC-NS-U-WC-NS-ABLATION-ATTR",
        
        # 备用：简化版消融属性互补（只保留强互补∩替补¬）
        # 一阶：强互补∩替补¬（基础结构）
        (("e", ("r",)), ("e", ("r", 'n'))): "ABLATION-ATTR-1ORDER-SC-NS",
        
        # 消融属性互补的二阶查询：两个一阶查询的交集
        ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n')))): "ABLATION-ATTR-2ORDER-SC-NS-I-SC-NS",
        
        # 消融属性互补的三阶查询：三个一阶查询的交集
        ((("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n'))), (("e", ("r",)), ("e", ("r", 'n')))): "ABLATION-ATTR-3ORDER-SC-NS-I-SC-NS-I-SC-NS",
        
        # 2. 消融替补过滤的查询结构
        # 消融后结构: 强互补 ∪ (弱互补 ∪ 属性互补)
        
        # 一阶：强互补 ∪ (弱互补 ∪ 属性互补)
        # = (("e", ("r",)), (("e", ("r",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ("u",))
        (
            ("e", ("r",)),  # SC
            (("e", ("r",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)),  # WC∪ATTR
            ("u",)
        ): "SC-U-WC-ATTR-U-ABLATION-SUB",
        
        # 二阶：两个消融替补过滤一阶查询的交集
        (
            (("e", ("r",)), (("e", ("r",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ("u",)),
            (("e", ("r",)), (("e", ("r",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ("u",))
        ): "SC-U-WC-ATTR-U-I-SC-U-WC-ATTR-U-ABLATION-SUB",
        
        # 三阶：三个消融替补过滤一阶查询的交集
        (
            (("e", ("r",)), (("e", ("r",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ("u",)),
            (("e", ("r",)), (("e", ("r",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ("u",)),
            (("e", ("r",)), (("e", ("r",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ("u",))
        ): "SC-U-WC-ATTR-U-I-SC-U-WC-ATTR-U-I-SC-U-WC-ATTR-U-ABLATION-SUB",
        
        # 备用：旧版消融替补过滤结构（保留兼容性）
        # 一阶：(强互补 ∪ 弱互补) ∪ (供应商互补 ∪ 语义互补)
        ((("e", ("r",)), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)): "ABLATION-SUB-1ORDER-SC-U-WC-U-VENDOR-U-SEMANTIC",
        
        # 消融替补过滤的二阶查询：两个一阶查询的交集
        (((("e", ("r",)), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ((("e", ("r",)), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",))): "ABLATION-SUB-2ORDER-SC-U-WC-U-ATTR-I-SC-U-WC-U-ATTR",
        
        # 消融替补过滤的三阶查询：三个一阶查询的交集
        (((("e", ("r",)), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ((("e", ("r",)), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",)), ((("e", ("r",)), ("e", ("r",)), ("u",)), (("e", ("r",)), ("e", ("r",)), ("u",)), ("u",))): "ABLATION-SUB-3ORDER-SC-U-WC-U-ATTR-I-SC-U-WC-U-ATTR-I-SC-U-WC-U-ATTR",
        
    }
    def load_pickle(self, path, query_types=None, union_type="DNF", verbose=0, num_relation=5):
        # 添加num_relation参数，默认为4以保持向后兼容性

        query_types = query_types or self.struct2type.values()
        new_query_types = []
        for query_type in query_types:
            if "u" in query_type:
                if "-" not in query_type:
                    query_type = "%s-%s" % (query_type, union_type)
                elif query_type[query_type.find("-") + 1:] != union_type:
                    continue
            new_query_types.append(query_type)
        self.id2type = sorted(new_query_types)
        self.type2id = {t: i for i, t in enumerate(self.id2type)}

        with open(os.path.join(path, "id2ent.pkl"), "rb") as fin:
            entity_vocab = pickle.load(fin)
        with open(os.path.join(path, "id2rel.pkl"), "rb") as fin:
            relation_vocab = pickle.load(fin)
        triplets = []
        num_samples = []
        for split in ["train", "valid", "test"]:
            triplet_file = os.path.join(path, "%s.txt" % split)
            with open(triplet_file) as fin:
                if verbose:
                    fin = tqdm(fin, "Loading %s" % triplet_file, utils.get_line_count(triplet_file))
                num_sample = 0
                for line in fin:
                    h, r, t = [int(x) for x in line.split()]
                    triplets.append((h, t, r))
                    num_sample += 1
                num_samples.append(num_sample)
        # 使用指定的num_relation参数
        self.load_triplet(triplets, entity_vocab=entity_vocab, relation_vocab=relation_vocab)
        # 在图创建后手动设置关系数量
        self.graph.num_relation = num_relation
        fact_mask = torch.arange(num_samples[0])
        self.fact_graph = self.graph.edge_mask(fact_mask)
        queries = []
        types = []
        answers = []
        num_samples = []
        max_query_length = 0

        for split in ["train", "valid", "test"]:
            if verbose:
                pbar = tqdm(desc="Loading %s-*.pkl" % split, total=3)
            with open(os.path.join(path, "%s-queries.pkl" % split), "rb") as fin:
                struct2queries = pickle.load(fin)
            if verbose:
                pbar.update(1)
            
            # 检查是否有新的查询结构类型，需要添加到struct2type字典
            unknown_structs = []
            for struct in struct2queries:
                if struct not in self.struct2type:
                    # 如果发现未知的查询结构，打印出来并添加到struct2type字典中
                    if verbose:
                        print(f"发现未知查询结构: {struct}")
                    # 生成标准化的结构名 - 改进的消融查询结构识别
                    if len(struct) == 3 and struct[2] == ('u',):
                        # 一阶查询结构
                        type_name = f"ABLATION-1ORDER-{len(unknown_structs)}"
                    elif len(struct) == 2:
                        # 二阶查询结构
                        type_name = f"ABLATION-2ORDER-{len(unknown_structs)}"
                    elif len(struct) == 3:
                        # 三阶查询结构
                        type_name = f"ABLATION-3ORDER-{len(unknown_structs)}"
                    elif len(struct) > 3:
                        # 更复杂的结构
                        type_name = f"ABLATION-COMPLEX-{len(struct)}D-{len(unknown_structs)}"
                    else:
                        # 其他情况使用通用命名
                        type_name = f"ABLATION-UNKNOWN-{len(unknown_structs)}"
                    unknown_structs.append((struct, type_name))
                    self.struct2type[struct] = type_name
            
            # 转换查询结构到类型
            try:
                type2queries = {self.struct2type[k]: v for k, v in struct2queries.items()}
                type2queries = {k: v for k, v in type2queries.items() if k in self.type2id}
            except KeyError as e:
                if verbose:
                    print(f"KeyError: 无法找到查询结构的映射: {e}")
                raise
                
            if split == "train":
                with open(os.path.join(path, "%s-answers.pkl" % split), "rb") as fin:
                    query2_answers = pickle.load(fin)
                if verbose:
                    pbar.update(2)
            else:
                with open(os.path.join(path, "%s-answers.pkl" % split), "rb") as fin:
                    query2_answers = pickle.load(fin)
                if verbose:
                    pbar.update(1)
            num_sample = sum([len(q) for t, q in type2queries.items()])
            if verbose:
                pbar = tqdm(desc="Processing %s queries" % split, total=num_sample)
            for type in type2queries:
                struct_queries = sorted(type2queries[type])
                for query in struct_queries:
                    answers.append(query2_answers[query])
                    query = Query.from_nested(query)
                    queries.append(query)
                    max_query_length = max(max_query_length, len(query))
                    types.append(self.type2id[type])
                    if verbose:
                        pbar.update(1)
            num_samples.append(num_sample)
# 这里把所有的查询全部转为后缀表达式的形式，这里就处理完成了
        self.queries = queries
        self.types = types
        self.answers = answers
        self.num_samples = num_samples
        self.max_query_length = max_query_length

    def __getitem__(self, index):
        query = self.queries[index]
        answer = torch.tensor(list(self.answers[index]), dtype=torch.long)
        return {
            "query": F.pad(query, (0, self.max_query_length - len(query)), value=query.stop),
            "type": self.types[index],
            "answer": functional.as_mask(answer, self.num_entity),
        }

    def __len__(self):
        return len(self.queries)

    def __repr__(self):
        lines = [
            "#entity: %d" % self.num_entity,
            "#relation: %d" % self.num_relation,
            "#triplet: %d" % self.num_triplet,
            "#query: %d" % len(self.queries),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits





@R.register("dataset.order-base")
class APILogicalQuery(LogicalQueryDataset):
    def __init__(self, path, query_types=None, union_type="DNF", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        path = os.path.join(path)
        
        # 根据数据集类型自动设置num_relation
        # HGA数据集：5个关系 (0-强互补, 1-弱互补, 2-替补, 3-vendor互补, 4-semantic互补)
        # PWA数据集：5个关系 (0-强互补, 1-弱互补, 2-替补, 3-vendor互补, 4-semantic互补)
        if "HGA" in path.upper() or "HUAWEI" in path.upper():
            num_relation = 5
            if verbose:
                print(f"检测到HGA数据集，设置num_relation=5")
        else:
            num_relation = 5
            if verbose:
                print(f"检测到PWA数据集，设置num_relation=5")
        
        self.load_pickle(path, query_types, union_type, verbose=verbose, num_relation=num_relation)


