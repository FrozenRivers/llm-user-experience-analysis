# 基于文本分析的生成式AI产品用户体验研究

## 项目简介
本项目为本科毕业论文配套代码，实现基于用户评论数据的生成式AI产品用户体验分析。

方法包括：
- BGE语义嵌入
- UMAP降维
- HDBSCAN聚类
- BERTopic主题建模
- RoBERTa情感分析
- 多元线性回归

## 数据说明
数据来源于应用商店评论（七麦数据），因版权限制未公开原始数据。

仓库提供示例数据用于流程复现。

## 方法流程
Raw → 清洗 → 嵌入 → 降维 → 聚类 → 主题 → 情感 → 回归

## 使用方法
```bash
pip install -r requirements.txt
python pipeline.py