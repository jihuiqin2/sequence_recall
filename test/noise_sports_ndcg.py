import matplotlib.pyplot as plt
import numpy as np

# 两个元素的数据和类别
categories = ['5', '10', '15', '20', '25', '30']
# values_element1 = [0.0230, 0.0229, 0.0222, 0.0206, 0.0188, 0.0178]
values_element2 = [0.0238, 0.0234, 0.0226, 0.0206, 0.018, 0.0173]
values_element3 = [0.0278, 0.0273, 0.0265, 0.0245, 0.0212, 0.0208]

# 设置条形图的宽度
bar_width = 0.3

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
ax.set_title('Sports')
ax.set_xlabel('Noise Ratio(%)')
ax.set_ylabel('NDCG@10')

yticks = [0.004, 0.008, 0.012, 0.016, 0.02, 0.024, 0.028, 0.032]  # 0.04
yticklabels = ['', '0.008', '0.012', '0.016', '0.02', '0.024', '0.028', '']  # 刻度标签，将底部的刻度标签设为空字符串
ax.set_yticks(yticks)
ax.set_ylim(yticks[0], yticks[-1])
ax.set_yticklabels(yticklabels)

# 显示网格线
ax.yaxis.grid(True, linewidth=0.2, linestyle='-', color='#cbcbcb')

# 添加图例
ax.legend()

# 调整左右两侧的空白边
plt.subplots_adjust(left=0.1, right=0.99)  # 根据需要调整左右边界的值
plt.subplots_adjust(top=0.95, bottom=0.08)

# 显示图形
plt.show()
