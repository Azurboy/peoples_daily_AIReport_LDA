import json
import numpy as np
import matplotlib.pyplot as plt
from gensim import corpora, models
from gensim.models import CoherenceModel
import seaborn as sns
from matplotlib import font_manager
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # macOS系统可用的中文字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
plt.style.use('seaborn-v0_8')

# 读取数据
def load_data(years):
    all_docs = []
    for year in years:
        with open(f'../../data/tokenized_data/2020-2022/tokenized_{year}.json', 'r', encoding='utf-8') as f:
            docs = json.load(f)
            all_docs.extend(docs)
    return all_docs

# 准备词典和语料库
def prepare_corpus(documents):
    # 定义要过滤的标点符号和常见停用词
    stopwords = ['，', '。', '、', '；', '：', '？', '！', '"', '"', ''', ''', '（', '）', 
                '《', '》', '【', '】', '-', '.', '的', '了', '和', '与', '或', '及', 
                '在', '是', '有', '为', '被', '所', '这', '那', '从', '对', '把', '将',
                '等', '中', '上', '下', '内', '外', '前', '后', '里', '通过', '以',
                '能', '要', '会', '到', '让', '使', '向', '并', '且', '而', '但',
                '地', '得', '着', '于', '由', '因', '为了', '一个', '没有', '可以']
    
    # 过滤文档中的停用词
    filtered_docs = []
    for doc in documents:
        filtered_doc = [word for word in doc if word not in stopwords and len(word) > 1]
        filtered_docs.append(filtered_doc)
    
    # 创建词典并过滤极低频和极高频词
    dictionary = corpora.Dictionary(filtered_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in filtered_docs]
    return dictionary, corpus

# 计算困惑度和一致性
def compute_metrics(corpus, dictionary, texts, start=2, limit=20, step=1):
    perplexity_values = []
    coherence_values = []
    model_list = []
    
    for num_topics in range(start, limit, step):
        # 训练LDA模型
        model = models.LdaModel(corpus=corpus,
                              num_topics=num_topics,
                              id2word=dictionary,
                              random_state=42,
                              passes=20)
        
        # 计算困惑度
        perplexity = model.log_perplexity(corpus)
        perplexity_values.append(perplexity)
        
        # 计算一致性
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        coherence_values.append(coherence)
        
        model_list.append(model)
        
        print(f'Topics: {num_topics}, Perplexity: {perplexity:.2f}, Coherence: {coherence:.4f}')
    
    return model_list, perplexity_values, coherence_values

# 可视化评估指标
def plot_metrics(perplexity_values, coherence_values, start=2, limit=11, step=1):
    x = range(start, limit, step)
    
    # 确保中文字体设置
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建双y轴图
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    
    # 绘制困惑度
    line1 = ax1.plot(x, perplexity_values, 'b-o', linewidth=2, markersize=8, label='Perplexity')
    ax1.set_xlabel('主题数', fontsize=14)
    ax1.set_ylabel('Perplexity', color='b', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='b', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制一致性
    line2 = ax2.plot(x, coherence_values, 'r-^', linewidth=2, markersize=8, label='Coherence')
    ax2.set_ylabel('Coherence', color='r', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=12)
    
    # 在主题数6处画垂直虚线，这里的最佳主题数是人工确定的。
    plt.axvline(x=6, color='r', linestyle='--', alpha=0.7)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc='best', fontsize=12)
    
    plt.title('主题数与模型评估指标关系 (2020-2022)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('../../data/tokenized_data/2020-2022/lda_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()



# 展示主题词
def display_topics(model, dictionary, num_words=10):
    topics = {}
    for topic_id in range(model.num_topics):
        topic_words = model.show_topic(topic_id, num_words)
        topics[topic_id] = [(word, round(prob, 4)) for word, prob in topic_words]
    
    return topics

# 添加统计主题文章数量的函数
def count_topic_documents(model, corpus):
    # 初始化主题计数器
    topic_counts = [0] * model.num_topics
    
    # 对每个文档，找出其主要主题
    for doc in corpus:
        # 获取文档的主题分布
        topic_dist = model.get_document_topics(doc)
        # 找出概率最大的主题
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
        # 对应主题计数加1
        topic_counts[dominant_topic] += 1
    
    return topic_counts

def main():
    # 加载2023-2024年的数据
    years = ['2020', '2022']
    documents = load_data(years)
    print(f"加载了{len(documents)}篇文章")
    
    # 准备语料库
    dictionary, corpus = prepare_corpus(documents)
    print(f"词典大小: {len(dictionary)}")
    
    # 计算不同主题数的评估指标
    start, limit, step = 2, 11, 1  # 修改为2-10的范围
    model_list, perplexity_values, coherence_values = compute_metrics(
        corpus, dictionary, documents, start=start, limit=limit, step=step)
    
    # 可视化结果
    plot_metrics(perplexity_values, coherence_values, start=start, limit=limit, step=step)
    
    # 使用6个主题训练最终模型
    final_model = models.LdaModel(
        corpus=corpus,
        num_topics=6,
        id2word=dictionary,
        random_state=42,
        passes=30
    )
    
    # 展示不同主题的词汇组合
    topics = display_topics(final_model, dictionary, num_words=15)
    
    # 统计每个主题的文章数量
    topic_counts = count_topic_documents(final_model, corpus)
    
    # 输出结果
    print("\n各主题的词汇组合和文章分布:")
    for topic_id in range(6):
        print(f"\n主题 {topic_id+1}:")
        print(f"文章数量: {topic_counts[topic_id]} (占比: {topic_counts[topic_id]/len(corpus):.2%})")
        print("关键词: " + ", ".join([f"{word} ({prob})" for word, prob in topics[topic_id]]))
    
    # 保存结果到文件
    with open('../../data/tokenized_data/202-2022/lda_topics.txt', 'w', encoding='utf-8') as f:
        f.write(f"主题数: 6\n\n")
        for topic_id in range(6):
            f.write(f"主题 {topic_id+1}:\n")
            f.write(f"文章数量: {topic_counts[topic_id]} (占比: {topic_counts[topic_id]/len(corpus):.2%})\n")
            f.write("关键词: " + ", ".join([f"{word} ({prob})" for word, prob in topics[topic_id]]))
            f.write("\n\n")
    
    # 可视化主题分布
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 7), topic_counts)
    plt.xlabel('主题编号')
    plt.ylabel('文章数量')
    plt.title('各主题文章数量分布')
    plt.xticks(range(1, 7))
    
    # 在柱状图上标注具体数值
    for i, count in enumerate(topic_counts):
        plt.text(i+1, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../../data/tokenized_data/2020-2022/topic_distribution.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()