import matplotlib.pyplot as plt
import numpy as np

# 创建一些示例数据
x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
y1 = np.array([0.0737, 0.0757, 0.0759, 0.0767, 0.0777, 0.0782, 0.0768, 0.0771, 0.0774, 0.0775, 0.0771])
y2 = np.array([0.1043, 0.1073, 0.1095, 0.1105, 0.1106, 0.1116, 0.110, 0.110, 0.1102, 0.1101, 0.1082])
y3 = np.array([0.1462, 0.1501, 0.1510, 0.1517, 0.1521, 0.1532, 0.151, 0.1514, 0.1518, 0.1514, 0.1486])

# 创建一个图形和坐标轴
fig, ax = plt.subplots()

# 绘制三条折线图
line1, = ax.plot(x, y1, marker='h', label='HR@5', color='#76b070')
line2, = ax.plot(x, y2, marker='h', label='HR@10', color='#f3b4c7')
line3, = ax.plot(x, y3, marker='h', label='HR@20', color='#90aadb')

# 绘制数据点
ax.scatter(x, y1, color='#76b070', zorder=10)
ax.scatter(x, y2, color='#f3b4c7', zorder=10)
ax.scatter(x, y3, color='#90aadb', zorder=10)

# 设置背景颜色为浅灰色
ax.axhspan(ymin=0, ymax=100, facecolor='#cbcbcb', alpha=0.5)  # 设置背景区域的范围和颜色

# 将刻度向内对齐
ax.tick_params(direction='in')

# 添加标题和标签
ax.set_title('Beauty')
ax.set_xlabel('λ')
# ax.set_ylabel('Y-axis')

# x轴刻度显示
xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
yticks = [0.044, 0.064, 0.084, 0.104, 0.124, 0.144, 0.164, 0.184]  # 0.02
yticklabels = ['', '0.064', '0.084', '0.104', '0.124', '0.144', '0.164', '']  # 刻度标签，将底部的刻度标签设为空字符串

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
