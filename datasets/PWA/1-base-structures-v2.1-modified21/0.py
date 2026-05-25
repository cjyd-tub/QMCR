import pickle
import random

path1 = "train-answers.pkl"
f1 = open(path1,'rb')
train_answers = pickle.load(f1)

path1 = "train-queries.pkl"
f1 = open(path1,'rb')
train_queries = pickle.load(f1)

path1 = "valid-answers.pkl"
f1 = open(path1,'rb')
valid_answers = pickle.load(f1)

path1 = "valid-queries.pkl"
f1 = open(path1,'rb')
valid_queries = pickle.load(f1)

path1 = "test-answers.pkl"
f1 = open(path1,'rb')
test_answers = pickle.load(f1)

path1 = "test-queries.pkl"
f1 = open(path1,'rb')
test_queries = pickle.load(f1)

# 输出查询结构类型统计
print("\n===== 查询结构类型统计 =====")
query_type_count = {}
for structure_type in train_queries:
    query_type_count[structure_type] = len(train_queries[structure_type])
    print(f"结构类型: {structure_type}, 查询数量: {len(train_queries[structure_type])}")

# 定义我们的一阶结构
sc_ns_wc_u_type = ((("e", ("r",)), ("e", ("r", 'n'))), ("e", ("r",)), ("u",))

# 统计一阶SC-NS-WC-U查询数量
one_order_count = 0
if sc_ns_wc_u_type in train_queries:
    one_order_count = len(train_queries[sc_ns_wc_u_type])
else:
    print("错误：没有找到SC-NS-WC-U结构查询！")

print(f"一阶SC-NS-WC-U查询数量: {one_order_count}")

# 关系统计
relation_counts = {}
for q in train_answers:
    if isinstance(q[0], int):
        # 基本关系
        rel = q[1][0]
        if rel not in relation_counts:
            relation_counts[rel] = 0
        relation_counts[rel] += 1

print("\n===== 基本关系统计 =====")
for rel, count in relation_counts.items():
    print(f"关系ID {rel}: {count} 个查询")

# 生成训练文件（转换为模型可用的格式）
def generate_data_file(queries, answers, filename, sc_ns_wc_u_relation_id=3):
    list_data = []
    sc_ns_wc_u_samples = 0
    basic_relation_samples = 0
    other_samples = 0
    api_counts = set()
    
    for q in answers:
        if isinstance(q[0], int):
            # 基础关系: 强互补(0), 弱互补(1), 替补(2)
            api = q[0]
            api_counts.add(api)
            ans = answers[q]
            rel = q[1][0]  # 关系ID
            for v in ans:
                list_data.append([api, rel, v])
                basic_relation_samples += 1
        else:
            # 复杂查询
            struct_type = None
            for type_struct in queries:
                if q in queries[type_struct]:
                    struct_type = type_struct
                    break
            
            if struct_type == sc_ns_wc_u_type:
                # 一阶SC-NS-WC-U：((强互补交替补取反)并弱互补)
                api = q[0][0][0]
                api_counts.add(api)
                rel = sc_ns_wc_u_relation_id  # 使用ID 3表示SC-NS-WC-U结构
                for v in answers[q]:
                    list_data.append([api, rel, v])
                    sc_ns_wc_u_samples += 1
            else:
                # 其他复杂结构
                other_samples += 1
    
    # 随机打乱数据
    random.shuffle(list_data)
    
    print(f"\n===== {filename} 统计 =====")
    print(f"生成样本总数: {len(list_data)}")
    print(f"基本关系样本数: {basic_relation_samples}")
    print(f"SC-NS-WC-U样本数: {sc_ns_wc_u_samples}")
    print(f"其他复杂关系样本数: {other_samples}")
    print(f"涉及的API数量: {len(api_counts)}")
    
    # 检查生成数据中的关系ID
    rel_id_counts = {}
    for sample in list_data:
        rel_id = sample[1]
        if rel_id not in rel_id_counts:
            rel_id_counts[rel_id] = 0
        rel_id_counts[rel_id] += 1
    
    print(f"\n===== {filename} 关系ID统计 =====")
    for rel_id, count in sorted(rel_id_counts.items()):
        print(f"关系ID {rel_id}: {count} 个三元组")
    
    # 保存为训练文件格式
    with open(filename, "w") as file:
        for line in list_data:
            line_str = "\t".join(map(str, line))
            file.write(line_str + "\n")
    
    print(f"文件已保存为: {filename}")
    return list_data

# 生成训练、验证和测试文件
train_data = generate_data_file(train_queries, train_answers, "train.txt")
valid_data = generate_data_file(valid_queries, valid_answers, "valid.txt")
test_data = generate_data_file(test_queries, test_answers, "test.txt")

# 创建id2rel.pkl文件，确保关系ID定义正确
# 强互补(0), 弱互补(1), 替补(2), SC-NS-WC-U(3)
id2rel = {0: "强互补", 1: "弱互补", 2: "替补", 3: "SC-NS-WC-U"}
with open("id2rel.pkl", "wb") as f:
    pickle.dump(id2rel, f)
print("\n关系ID映射已保存：")
print(id2rel)