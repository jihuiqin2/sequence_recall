import matplotlib.pyplot as plt
import numpy as np

# 创建一些示例数据
x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
y1 = np.array([0.0519, 0.0581, 0.0531, 0.0534, 0.0545])

# 创建一个图形和坐标轴
fig, ax = plt.subplots()

# 绘制三条折线图
line1, = ax.plot(x, y1, marker='h', label='HR@5', color='#5eadc2')

# 绘制数据点
ax.scatter(x, y1, color='#5eadc2', zorder=10, alpha=1)

# 将刻度向内对齐
ax.tick_params(direction='in')

# 添加标题和标签
ax.set_title('Sports')
ax.set_xlabel('λ')
# ax.set_ylabel('Y-axis')

# x轴刻度显示
xticks = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
yticks = [0.0502, 0.0517, 0.0532, 0.0547, 0.0562, 0.0577, 0.0592, 0.0607]  # 0.0015
yticklabels = ['', '0.0517', '0.0532', '0.0547', '0.0562', '0.0577', '0.0592', '']  # 刻度标签，将底部的刻度标签设为空字符串

ax.set_xticks(xticks)
ax.set_xlim(xticks[0], xticks[-1])  # 调整 x 轴范围，使开始刻度对齐最左侧

ax.set_yticks(yticks)
ax.set_ylim(yticks[0], yticks[-1])
ax.set_yticklabels(yticklabels)

# 移除右侧和上侧的边框线
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# 显示网格线
ax.grid(True, which='major', linestyle='-', color='#cbcbcb')

# 调整左右两侧的空白边
plt.subplots_adjust(left=0.09, right=0.97)  # 根据需要调整左右边界的值
plt.subplots_adjust(top=0.95, bottom=0.08)

# 显示图形
plt.show()
