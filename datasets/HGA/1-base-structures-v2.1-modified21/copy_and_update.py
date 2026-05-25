#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复制生成的查询结构数据到HGA数据集目录，并更新关系映射
"""

import pickle
import shutil
import os

# 源目录
src_dir = "../../../../data_processing/hg/edition15/1-2-3-order/1-base-structures-v2.1-modified21"
# 当前目录（目标目录）
dst_dir = "."

# 1. 复制pkl文件
files_to_copy = [
    "train-queries.pkl",
    "train-answers.pkl", 
    "valid-queries.pkl",
    "valid-answers.pkl",
    "test-queries.pkl",
    "test-answers.pkl",
]

print("📦 复制生成的数据文件...")
for f in files_to_copy:
    src = os.path.join(src_dir, f)
    dst = os.path.join(dst_dir, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  ✅ {f}")
    else:
        print(f"  ❌ {f} 不存在")

# 2. 更新 rel2id.pkl (5个关系)
rel2id = {
    0: 0,  # 强互补 (Strong Complementary)
    1: 1,  # 弱互补 (Weak Complementary)
    2: 2,  # 替补 (Substitute)
    3: 3,  # 供应商互补 (Vendor Complementary)
    4: 4,  # 语义互补 (Semantic Complementary)
}

with open('rel2id.pkl', 'wb') as f:
    pickle.dump(rel2id, f)
print("✅ 已更新 rel2id.pkl (5个关系)")

# 3. 更新 id2rel.pkl
id2rel = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
}

with open('id2rel.pkl', 'wb') as f:
    pickle.dump(id2rel, f)
print("✅ 已更新 id2rel.pkl (5个关系)")

# 4. 更新 stats.txt
with open('stats.txt', 'w') as f:
    f.write('numentity: 1260\n')
    f.write('numrelations: 5\n')
print("✅ 已更新 stats.txt (numrelations: 5)")

# 5. 验证复制的数据
print("\n📊 验证数据...")
for f in files_to_copy:
    if os.path.exists(f):
        with open(f, 'rb') as fp:
            data = pickle.load(fp)
            if isinstance(data, dict):
                print(f"  {f}: {len(data)} 条")
            elif isinstance(data, list):
                print(f"  {f}: {len(data)} 条")

# 6. 显示查询结构类型
print("\n📐 查询结构类型:")
with open('train-queries.pkl', 'rb') as f:
    train_queries = pickle.load(f)
    for q_type in train_queries.keys():
        print(f"  {q_type}")

print("\n🎯 完成！HGA数据集已更新为V21查询结构")
