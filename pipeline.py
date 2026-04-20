# 按序执行完整分析流程

import importlib, sys

steps = [
    ("processing",  "数据清洗"),
    ("clustering",  "BERTopic 建模"),
    ("sentiment",   "情感打分"),
    ("dimension",   "维度映射"),
    ("coherence",   "一致性验证"),
    ("regression",  "回归分析"),
]

for module_name, desc in steps:
    print(f"\n{'='*40}")
    print(f"▶ 开始：{desc}（{module_name}.py）")
    print('='*40)
    module = importlib.import_module(module_name)
    importlib.reload(module)

print("\n✅ 全流程完成")