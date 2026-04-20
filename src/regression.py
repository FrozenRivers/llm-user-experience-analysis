import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from config import DIMENSION_OUT

# ================================
# 1. 读取数据
# ================================
df = pd.read_csv(DIMENSION_OUT)
df = df[df["维度"] != "未分类"]
df = df.rename(columns={"评级": "rating", "字数": "word_count", "维度": "dimension"})
df_model = df[["rating", "sentiment_score", "word_count", "dimension", "产品名称"]].dropna()

# ================================
# 2. VIF 多重共线性检验
# ================================
X = pd.get_dummies(df_model[['sentiment_score', 'dimension', 'word_count']], drop_first=True)
X = X.astype(float)
X = sm.add_constant(X)
X = X.replace([float('inf'), -float('inf')], pd.NA).dropna()

vif = pd.DataFrame()
vif["变量"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\n=== VIF 检验 ===")
print(vif)

# ================================
# 3. 全量 OLS（以"整体满意度与情绪化表达"为基准组）
# ================================
BASE_DIMENSION = "整体满意度与情绪化表达"

formula = (
    f"rating ~ sentiment_score + word_count + "
    f"C(dimension, Treatment(reference='{BASE_DIMENSION}'))"
)

model_all = smf.ols(formula, data=df_model).fit()
print("\n=== 全量回归结果 ===")
print(model_all.summary())

# ================================
# 4. 分产品 OLS
# ================================
print("\n=== 分产品回归结果 ===")
for product in df_model['产品名称'].unique():
    sub_df = df_model[df_model['产品名称'] == product]
    try:
        model = smf.ols(formula, data=sub_df).fit()
        print(f"\n===== {product} =====")
        print(model.params)
    except Exception as e:
        print(f"{product} 回归失败：{e}")