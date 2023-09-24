import matplotlib.pyplot as plt

# 数据
x = [1, 2, 3, 4, 5]
y = [10, 15, 7, 12, 9]

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 绘制柱状图
bars = ax.bar(x, y, color=['#808080', '#00008B', '#B22222', '#90EE90', '#FF69B4'])

# 设置y轴刻度从5开始
start_y = 5
yticks = list(range(start_y, max(y) + 1))  # 创建从5到最大值的刻度
ax.set_yticks(yticks)

plt.show()
