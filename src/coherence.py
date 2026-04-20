import pandas as pd
import jieba
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from config import DIMENSION_OUT

# ================================
# 1. 读取数据（排除未分类）
# ================================
df = pd.read_csv(DIMENSION_OUT)
df = df.dropna(subset=["full_text"])
df = df[df["维度"] != "未分类"]

docs = df["full_text"].astype(str).tolist()

# ================================
# 2. 分词 + 去停用词
# ================================
stopwords = set(["的","了","是","我","也","很","在","不","有","就"])

def tokenize(text):
    tokens = list(jieba.cut(text))
    return [w for w in tokens if w not in stopwords and len(w) > 1]

docs_tokenized = [tokenize(doc) for doc in docs]

# ================================
# 3. 构建词典
# ================================
dictionary = Dictionary(docs_tokenized)

# ================================
# 4. 用人工归维结果构建 topic word list
#    每个维度取高频词 Top 10，作为一个 topic
# ================================
from sklearn.feature_extraction.text import CountVectorizer

topics_words = []

for dim in df['维度'].unique():
    dim_docs = df[df['维度'] == dim]['full_text'].astype(str).tolist()
    if len(dim_docs) < 5:
        continue
    vec = CountVectorizer(max_features=10, tokenizer=tokenize, token_pattern=None)
    vec.fit(dim_docs)
    topics_words.append(list(vec.vocabulary_.keys()))

# ================================
# 5. 计算 Coherence（c_v）
# ================================
coherence_model = CoherenceModel(
    topics=topics_words,
    texts=docs_tokenized,
    dictionary=dictionary,
    coherence='c_v'
)

print("Coherence Score (c_v):", coherence_model.get_coherence())