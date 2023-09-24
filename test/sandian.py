import numpy as np
import matplotlib.pyplot as plt

# 创建示例数据
num_points = 10
x = np.random.randn(num_points)
y = np.random.randn(num_points)
freq = [1, 1, 1, 1, 10, 1, 1, 5, 1, 1]  # 频率信息
print(x, y, freq)

# 绘制散点图，根据频率信息进行颜色映射
plt.scatter(x, y, c=freq, cmap='viridis', s=50, alpha=0.7)

# 添加颜色条
color_bar = plt.colorbar()
color_bar.set_label('Frequency')

# 添加标题和轴标签
plt.title("Scatter Plot with Color Mapping")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# 控制x轴和y轴坐标范围
plt.xlim(-1, 1)  # 控制x轴坐标范围为-3到3
plt.ylim(-3, 3)  # 控制y轴坐标范围为-3到3

# 显示图表
plt.show()
