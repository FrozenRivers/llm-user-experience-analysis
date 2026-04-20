# ================================
# 全局配置：路径 / 模型 / 超参数
# ================================

# 路径
RAW_DATA_DIR = "./data/raw"
CLEANED_DATA = "./data/processed/cleaned_data.csv"
TOPIC_RAW = "./data/processed/LLM_Topic_Raw.csv"
SENTIMENT_OUT = "./data/processed/LLM_Sentiment.csv"
DIMENSION_OUT = "./data/processed/LLM_Sentiment_Dimension.csv"

# 模型
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
SENTIMENT_MODEL = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"

# BERTopic 超参数
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
HDBSCAN_MIN_CLUSTER_SIZE = 40
HDBSCAN_MIN_SAMPLES = 5
NGRAM_RANGE = (1, 2)
MIN_DF = 5