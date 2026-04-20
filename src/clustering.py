import pandas as pd
import jieba
import emoji
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from config import (CLEANED_DATA, TOPIC_RAW, EMBEDDING_MODEL,
                    UMAP_N_NEIGHBORS, UMAP_N_COMPONENTS,
                    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
                    NGRAM_RANGE, MIN_DF)

# ================================
# 1. 读取数据
# ================================
df = pd.read_csv(CLEANED_DATA)

# ================================
# 2. 分词配置
# ================================
user_dict = [
    "免单卡", "助力", "阿里", "豆包", "ds", "deepseek",
    "千问", "淘宝闪购", "活动免单", "虚假宣传", "服务器繁忙",
    "点不了", "验证码", "算力", "满血版", "降智", "智谱"
]
for word in user_dict:
    jieba.add_word(word)

custom_stopwords = [
    '这个', '那个', '还是', '可以', '觉得', '感觉',
    '真的', '一个', '结果', '元宝', '阿里', '豆包',
    'ds', 'deepseek', '千问'
]

def clean_and_tokenize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = emoji.demojize(text, delimiters=(" ", " "))
    words = jieba.lcut(text)
    return " ".join([w for w in words if len(w) > 1 and w not in custom_stopwords])

# ================================
# 3. 构建文档
# ================================
df['full_text'] = df['标题'].fillna('') + " " + df['内容'].fillna('')
docs = df['full_text'].apply(clean_and_tokenize).tolist()

# ================================
# 4. 模型配置
# ================================
vectorizer_model = CountVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF)

umap_model = UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    n_components=UMAP_N_COMPONENTS,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
    prediction_data=True
)

topic_model = BERTopic(
    embedding_model=EMBEDDING_MODEL,
    vectorizer_model=vectorizer_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    nr_topics=None,
    verbose=True
)

# ================================
# 5. 训练
# ================================
print("正在训练模型...")
topics, probs = topic_model.fit_transform(docs)

unique_topics = set(topics) - {-1}
print(f"\n当前主题数量（不含噪声 -1）：{len(unique_topics)}")

# ================================
# 6. 输出
# ================================
df['Topic'] = topics
df['full_text'] = df['full_text']  # 保留供后续使用
df.to_csv(TOPIC_RAW, index=False, encoding='utf-8-sig')
print("完成！输出：", TOPIC_RAW)