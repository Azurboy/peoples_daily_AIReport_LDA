import pandas as pd
import numpy as np
from datetime import datetime

# 读取并统计2017-2024年的数据
years = range(2017, 2025)
annual_stats = []
early_year_stats = []

for year in years:
    try:
        # 读取Excel文件
        df = pd.read_excel(f'data/new_data/new{year}.xlsx')
        
        # 将日期字符串转换为datetime格式
        try:
            df['date'] = pd.to_datetime(df['date'].str.replace('.', '-'))
        except:
            # 如果日期转换失败，跳过转换步骤，继续使用原始日期
            pass
        
        # 计算年度总报道数
        total_articles = len(df)
        
        # 计算AI相关报道数（忽略空格）
        ai_articles = df[df['text'].str.contains('人工智能', na=False)].shape[0]
        
        # 添加到年度统计
        annual_stats.append({'year': year, 'total': total_articles, 'ai': ai_articles})
        
        # 统计每年2月15日之前的数据
        early_date = f'{year}-02-15'
        early_df = df[df['date'] <= early_date]
        early_total = len(early_df)
        early_ai = early_df[early_df['text'].str.contains('人工智能', na=False)].shape[0]
        
        early_year_stats.append({'year': year, 'early_total': early_total, 'early_ai': early_ai})
    
    except Exception as e:
        print(f"处理{year}年数据时出错：{str(e)}")
        continue

# 转换为DataFrame
annual_df = pd.DataFrame(annual_stats)
early_df = pd.DataFrame(early_year_stats)

# 添加2025年已知数据
early_df = pd.concat([early_df, pd.DataFrame([{
    'year': 2025, 
    'early_total': 4992, 
    'early_ai': 365
}])], ignore_index=True)

# 计算比例
# 计算比值关系
annual_df['ratio'] = annual_df['ai'] / annual_df['total']  # 全年比值
early_df['ratio'] = early_df['early_ai'] / early_df['early_total']  # 早期比值

# 分析早期比值和全年比值的关系（只使用2020-2024年的数据）
ratios_data = pd.DataFrame({
    'year': annual_df[annual_df['year'] >= 2020]['year'],
    'early_ratio': early_df[(early_df['year'] >= 2020) & (early_df['year'] < 2025)]['ratio'].values,
    'annual_ratio': annual_df[annual_df['year'] >= 2020]['ratio'].values
})

# 计算比值关系的年度变化
ratios_data['ratio_factor'] = ratios_data['annual_ratio'] / ratios_data['early_ratio']

# 使用最近三年的平均比值因子
recent_factors = ratios_data['ratio_factor'].tail(3).mean()

# 预测2025年全年比值
early_ratio_2025 = 365 / 4992  # 2025年早期比值
predicted_ratio_2025 = early_ratio_2025 * recent_factors

# 预测2025年全年数据
predicted_total_2025 = 4992 * (early_df[early_df['year'] == 2024].iloc[0]['early_total'] / early_df[early_df['year'] == 2025].iloc[0]['early_total'])
predicted_ai_2025 = int(predicted_total_2025 * predicted_ratio_2025)
predicted_total_2025 = int(predicted_total_2025)

print(f"预计2025年AI报道占比：{predicted_ratio_2025:.4%}")