import pandas as pd
import json
from tqdm import tqdm
import time
import os
from openai import OpenAI

def process_and_save_documents():
    client = OpenAI(
        api_key="你自己的API",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    df = pd.read_excel(r'data/new_data/new2024.xlsx')
    # 添加缺失值处理
    df_filtered = df[df['text'].str.contains('人工智能', na=False)]  # 处理NaN值
    
    output_file = r'data/new_data/tokenized_2024.json'  # 定义输出文件路径
    save_interval = 5  # 每处理5个文档保存一次
    
    # 加载已有结果
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = []
    
    
    try:
        for i, text in enumerate(tqdm(df_filtered['text'].tolist()[start_index:]), start_index + 1):
            try:
                prompt = f"""请对以下文本进行智能分词和过滤，要求：
                1. 重点识别人工智能、科技创新相关的专业术语和重要概念
                2. 保持专业术语的完整性（如"人工智能"、"机器学习"等）
                3. 去除无实际意义的词语
                4. 分词结果用空格分隔
                5. 只返回分词结果，不要其他解释
                
                文本：{text}"""
                
                response = client.chat.completions.create(
                    model="qwen-long",  # 使用通义千问长文本模型（主要是便宜）
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # 响应结构保持与OpenAI兼容
                if response.choices:
                    result = response.choices[0].message.content.strip()
                    results.append(result.split())
                
                # 添加定期保存机制（每5篇保存一次）
                if i % save_interval == 0:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"\n* 自动保存前 {i} 篇文档结果")
                time.sleep(1)
                
            except Exception as e:
                print(f"\n文档 {i}: 处理失败")
                print(f"错误信息: {str(e)}")
                continue
    except KeyboardInterrupt:
        print("\n捕获到中断信号，正在保存已处理结果...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"已成功保存 {len(results)} 篇文档至 {output_file}")
        exit()
    
    with open(r'data/new_data/tokenized_2024.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存至 data/new_data/tokenized_2024.json")

if __name__ == "__main__":
    process_and_save_documents()