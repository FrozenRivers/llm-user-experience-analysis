import pandas as pd
import os
from config import RAW_DATA_DIR, CLEANED_DATA

# ================================
# 1. 读取所有 xlsx，统一加产品名列
# ================================
all_data = []

for file in os.listdir(RAW_DATA_DIR):
    if file.lower().endswith('.xlsx'):
        filepath = os.path.join(RAW_DATA_DIR, file)
        df = pd.read_excel(filepath)
        product_name = file.split('_')[0]
        df['产品名称'] = product_name
        all_data.append(df)

# ================================
# 2. 合并
# ================================
df_all = pd.concat(all_data, ignore_index=True)
print("合并完成，总行数：", len(df_all))

# ================================
# 3. 清洗
# ================================
df_all = df_all.drop_duplicates(subset=['内容'], keep='first')
print("去重后行数：", len(df_all))

df_all = df_all[~df_all['内容'].str.contains('（该条评论已经被删除）', na=False)]
print("过滤已删除评论后行数：", len(df_all))

df_all = df_all[df_all['内容'].str.len() >= 4]
print("过滤短评论后行数：", len(df_all))

# ================================
# 4. 输出
# ================================
os.makedirs(os.path.dirname(CLEANED_DATA), exist_ok=True)
df_all.to_csv(CLEANED_DATA, index=False, encoding='utf-8-sig')
print("完成！输出：", CLEANED_DATA)

print("\n各产品评论数量：")
print(df_all['产品名称'].value_counts())