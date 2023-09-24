import matplotlib.pyplot as plt
import numpy as np

# 创建一些示例数据
x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
y1 = np.array([0.0328, 0.0338, 0.0348, 0.0353, 0.0356, 0.0367, 0.0361, 0.0346, 0.0336, 0.0339, 0.0321])
y2 = np.array([0.0501, 0.0519, 0.0529, 0.0556, 0.0566, 0.0576, 0.0569, 0.0546, 0.0541, 0.0545, 0.0516])
y3 = np.array([0.0742, 0.0772, 0.0782, 0.0821, 0.0833, 0.084, 0.0832, 0.0805, 0.0815, 0.0816, 0.0781])

# 创建一个图形和坐标轴
fig, ax = plt.subplots()

# 绘制三条折线图
line1, = ax.plot(x, y1, marker='h', label='HR@5', color='#76b070')
line2, = ax.plot(x, y2, marker='h', label='HR@10', color='#f3b4c7')
line3, = ax.plot(x, y3, marker='h', label='HR@20', color='#90aadb')

# 绘制数据点
ax.scatter(x, y1, color='#76b070', zorder=10, alpha=1)
ax.scatter(x, y2, color='#f3b4c7', zorder=10)
ax.scatter(x, y3, color='#90aadb', zorder=10)

# 设置背景颜色为浅灰色
ax.axhspan(ymin=0, ymax=100, facecolor='#cbcbcb', alpha=0.5)  # 设置背景区域的范围和颜色

# 将刻度向内对齐
ax.tick_params(direction='in')

# 添加标题和标签
ax.set_title('Sports')
ax.set_xlabel('λ')
# ax.set_ylabel('Y-axis')

# x轴刻度显示
xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
yticks = [0.026, 0.038, 0.05, 0.062, 0.074, 0.086, 0.098, 0.11]  # 0.02
yticklabels = ['', '0.038', '0.050', '0.062', '0.074', '0.086', '0.098', '']  # 刻度标签，将底部的刻度标签设为空字符串

ax.set_xticks(xticks)
ax.set_xlim(xticks[0], xticks[-1])  # 调整 x 轴范围，使开始刻度对齐最左侧

ax.set_yticks(yticks)
ax.set_ylim(yticks[0], yticks[-1])
ax.set_yticklabels(yticklabels)

# 移除右侧和上侧的边框线
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# 添加图例
legend = ax.legend(loc='upper right', edgecolor='#ffffff', shadow=False, bbox_to_anchor=(1.0, 1.0), handlelength=2,
                   handleheight=1)
legend.get_frame().set_facecolor('white')

# 显示网格线
ax.grid(True, which='major', linestyle='-', color='white')

# 调整左右两侧的空白边
plt.subplots_adjust(left=0.09, right=0.97)  # 根据需要调整左右边界的值
plt.subplots_adjust(top=0.95, bottom=0.08)

# 显示图形
plt.show()
