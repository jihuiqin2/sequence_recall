import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

x = np.array([0, 1, 2, 3])
y = np.array([0.0579, 0.0578, 0.0598, 0.0617])

# 创建一个图形和坐标轴对象
fig, ax = plt.subplots()

# 设置条形图的宽度
bar_width = 0.4

# [grey, darkblue, firebrick, lightgreen, hotpink]
colors = ['#c5e0b3', '#f1e3c2', '#9eb6cc', '#d79ec2']
labels = ['FDSRec w/o FA', 'FDSRec w/o FL.aug', 'FDSRec w/o SL', 'FDSRec']
bars1 = ax.bar(x, y, bar_width, zorder=10, color=colors)

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 垂直偏移量，可以根据需要调整
                textcoords="offset points",
                ha='center', va='bottom')

# 创建图例
legend_elements = [Patch(facecolor=color, label=label) for color, label in zip(colors, labels)]

# 同时隐藏刻度和坐标轴标签
ax.xaxis.set_major_locator(plt.NullLocator())

# 设置y轴刻度
yticks = [0.044, 0.048, 0.052, 0.056, 0.06, 0.064]  # 自定义刻度值 0.004
ax.set_yticks(yticks)
ax.set_ylim(yticks[0], yticks[-1])

# 添加标题和标签
ax.set_title('Beauty')
ax.set_ylabel('NDCG@10')

# 显示网格线
ax.yaxis.grid(True, linewidth=0.2, linestyle='-', color='#cbcbcb')

# 调整左右两侧的空白边
plt.subplots_adjust(left=0.12, right=0.99)  # 根据需要调整左右边界的值
plt.subplots_adjust(top=0.95, bottom=0.02)

# 添加图例
ax.legend(handles=legend_elements, loc="upper left")

plt.show()
