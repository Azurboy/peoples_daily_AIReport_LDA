import zhipuai
import pandas as pd
import json
from tqdm import tqdm
import time
import os

def load_progress():
    """加载已处理的进度"""
    progress_file = 'data/new_data/processing_progress.json'
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            return saved_data.get('processed_count', 0), saved_data.get('results', [])
    return 0, []

def save_progress(processed_count, results):
    """保存处理进度"""
    progress_file = 'data/new_data/processing_progress.json'
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump({
            'processed_count': processed_count,
            'results': results
        }, f, ensure_ascii=False, indent=2)

#需要替换API
def process_and_save_documents():
    client = zhipuai.ZhipuAI(api_key="这里需要替换成你自己的API")
    
    df = pd.read_excel(r'data/new_data/new2024.xlsx')
    df['text'] = df['text'].fillna('')
    df_filtered = df[df['text'].str.contains('人工智能', na=False)]
    
    # 加载已处理的进度
    processed_count, results = load_progress()
    
    print(f"=== 继续处理，已完成{processed_count}篇，共{len(df_filtered)}篇文档 ===")
    
    try:
        for i, text in enumerate(tqdm(df_filtered['text'].tolist()[processed_count:]), processed_count + 1):
            try:
                prompt = f"""请对以下文本进行智能分词和过滤，要求：
                1. 重点识别人工智能、科技创新相关的专业术语和重要概念
                2. 保持专业术语的完整性（如"人工智能"、"机器学习"等）
                3. 去除无实际意义的词语
                4. 分词结果用空格分隔
                5. 只返回分词结果，不要其他解释
                
                文本：{text}"""
                
                response = client.chat.completions.create(
                    model="glm-4",#说实话别用GLM系列，4贵的要死，flash效果又不好，感觉不如Qwen
                    messages=[{"role": "user", "content": prompt}]
                )
                
                if response.choices:
                    result = response.choices[0].message.content.strip()
                    results.append(result.split())
                
                # 每处理10篇文档保存一次进度
                if i % 10 == 0:
                    save_progress(i, results)
                    time.sleep(1)
                    
            except Exception as e:
                print(f"\n文档 {i}: 处理失败")
                print(f"错误信息: {str(e)}")
                # 保存当前进度
                save_progress(i-1, results)
                continue
    
    except KeyboardInterrupt:
        print("\n检测到程序中断，正在保存进度...")
        save_progress(i-1, results)
        print(f"进度已保存，已处理 {i-1} 篇文档")
        return
    
    # 处理完成，保存最终结果
    with open(r'data/new_data/tokenized_2024.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 清理进度文件
    if os.path.exists('data/new_data/processing_progress.json'):
        os.remove('data/new_data/processing_progress.json')
    
    print(f"全部处理完成！结果已保存至 tokenized_2024.json")

if __name__ == "__main__":
    process_and_save_documents()