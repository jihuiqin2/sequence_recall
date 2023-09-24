import matplotlib.pyplot as plt
import numpy as np

# 两个元素的数据和类别
categories = ['25', '50', '75', '100']
# values_element1 = [0.0084, 0.0608, 0.0657, 0.0756]
values_element2 = [0.0112, 0.0288, 0.0532, 0.0781]
values_element3 = [0.0136, 0.0303, 0.0645, 0.1116]

# 设置条形图的宽度
bar_width = 0.2

# 创建一个图形和坐标轴对象
fig, ax = plt.subplots()

# 绘制两个元素的条形图
# bars1 = ax.bar(np.arange(len(categories)), values_element1, bar_width, label='MoCo4SRec', color='#00a6d9',
#                zorder=10)
bars2 = ax.bar(np.arange(len(categories)) + bar_width, values_element2, bar_width, label='CLF4SRec', color='#9eb6cc',
               zorder=10)
bars3 = ax.bar(np.arange(len(categories)) + bar_width + bar_width, values_element3, bar_width, label='FDSRec',
               color='#d79ec2', zorder=10)

# 添加类别标签和标题
ax.set_xticks(np.arange(len(categories)) + bar_width / 2)
ax.set_xticklabels(categories)

# 将刻度向内对齐
ax.tick_params(direction='in')

# 添加标题和标签
ax.set_title('Beauty')
ax.set_xlabel('Percentage of Training Data(%)')
ax.set_ylabel('HR@10')

yticks = [0.0, 0.018, 0.036, 0.054, 0.072, 0.09, 0.108, 0.126]  # 0.018
yticklabels = ['', '0.018', '0.036', '0.054', '0.072', '0.09', '0.108', '']  # 刻度标签，将底部的刻度标签设为空字符串
ax.set_yticks(yticks)
ax.set_ylim(yticks[0], yticks[-1])
ax.set_yticklabels(yticklabels)

# 显示网格线
ax.yaxis.grid(True, linewidth=0.2, linestyle='-', color='#cbcbcb')

# 添加图例
ax.legend(loc=2)

# 调整左右两侧的空白边
plt.subplots_adjust(left=0.11, right=0.99)  # 根据需要调整左右边界的值
plt.subplots_adjust(top=0.95, bottom=0.09)

# 显示图形
plt.show()
