import pandas as pd
from transformers import pipeline
from config import TOPIC_RAW, SENTIMENT_OUT, SENTIMENT_MODEL

# ================================
# 1. 加载模型
# ================================
print("正在加载情感模型...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL
)

# ================================
# 2. 读取数据
# ================================
df = pd.read_csv(TOPIC_RAW)

# ================================
# 3. 情感打分（连续 0~1）
# ================================
def get_sentiment_score(text):
    try:
        if not isinstance(text, str) or text.strip() == "":
            return 0.5
        result = sentiment_model(text[:512])[0]
        score = float(result['score'])
        if result['label'].lower() == 'positive':
            return 0.5 + 0.5 * score
        else:
            return 0.5 - 0.5 * score
    except:
        return 0.5

print("正在计算情感得分...")
df['sentiment_score'] = df['内容'].apply(get_sentiment_score)

# ================================
# 4. 字数（控制变量）
# ================================
df['字数'] = df['内容'].apply(lambda x: len(x) if isinstance(x, str) else 0)

# ================================
# 5. 描述统计
# ================================
print("\n=== 情感得分描述统计 ===")
print(df['sentiment_score'].describe())

# ================================
# 6. 输出
# ================================
df.to_csv(SENTIMENT_OUT, index=False, encoding='utf-8-sig')
print("完成！输出：", SENTIMENT_OUT)