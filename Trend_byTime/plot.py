import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 数据
years = range(2017, 2026)
ratios = [1.87, 3.16, 4.41, 4.70, 4.31, 4.34, 5.58, 7.14, 11.0780]

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制2017-2024的实线
plt.plot(years[:-1], ratios[:-1], 'b-', linewidth=2, marker='o', markersize=8, label='实际数据')

# 绘制2024-2025的虚线
plt.plot(years[-2:], ratios[-2:], 'b--', linewidth=2, marker='o', markersize=8, label='预测数据')

# 设置标题和标签
plt.title('2017-2025年《人民日报》人工智能相关报道占比变化', fontsize=14, pad=15)
plt.xlabel('年份', fontsize=12)
plt.ylabel('占比 (%)', fontsize=12)

# 设置x轴刻度
plt.xticks(years, rotation=0)

# 设置y轴范围和网格
plt.ylim(0, 12)
plt.grid(True, linestyle='--', alpha=0.7)

# 添加数据标签
for i, ratio in enumerate(ratios):
    plt.annotate(f'{ratio}%', 
                xy=(years[i], ratio),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                va='bottom')

# 添加图例
plt.legend(loc='upper left')

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('../data/new_data/ai_ratio_trend.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()