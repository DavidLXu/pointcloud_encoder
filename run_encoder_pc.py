import torch
import numpy as np
from PN_Model import AutoencoderPN

# 读取 PLY 点云文件
try:
    import open3d as o3d
    # 使用 open3d 读取 PLY 文件
    pcd = o3d.io.read_point_cloud('./pc_oneframe_20260117_155410.ply')
    points = np.asarray(pcd.points)  # 形状: [N, 3]
except ImportError:
    # 如果没有 open3d，尝试使用 plyfile
    try:
        from plyfile import PlyData
        plydata = PlyData.read('./pc_oneframe_20260117_155410.ply')
        vertex = plydata['vertex']
        points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)  # 形状: [N, 3]
    except ImportError:
        raise ImportError("请安装 open3d 或 plyfile 库来读取 PLY 文件: pip install open3d 或 pip install plyfile")

print(f"读取的点云包含 {points.shape[0]} 个点")

# 将点云中心移到原点并单位化
centroid = points.mean(axis=0)
points = points - centroid
max_dist = np.linalg.norm(points, axis=1).max()
if max_dist > 0:
    points = points / max_dist

# 将点云转换为模型期望的格式
num_points = 1024
if points.shape[0] > num_points:
    # 如果点数过多，随机采样
    indices = np.random.choice(points.shape[0], num_points, replace=False)
    points = points[indices]
elif points.shape[0] < num_points:
    # 如果点数不足，重复最后一个点进行填充
    padding = np.tile(points[-1:], (num_points - points.shape[0], 1))
    points = np.vstack([points, padding])

# 转换为 torch tensor 并调整形状: [1, 3, num_points]
points_tensor = torch.from_numpy(points).float().T.unsqueeze(0)  # [1, 3, 1024]

# 如果 tensor shape 是 [1, 1024, 3]，使用 .permute(0, 2, 1) 将其更改为 [1, 3, 1024]

# 加载模型
pointcloud_encoder = AutoencoderPN(k=128, num_points=1024)
pointcloud_encoder.load_state_dict(torch.load('./023980.pth', map_location='cpu'))
pointcloud_encoder.to('cuda')
pointcloud_encoder.eval()

# 将点云数据移到 GPU
points_tensor = points_tensor.to('cuda')


# 进行编码
with torch.no_grad():
    encoded_pc, restoration = pointcloud_encoder(points_tensor)

# 打印编码后的潜在向量
# encoded_pc 的形状是 [1, 128, 1]，我们需要将其压缩为 [128]
latent_vector = encoded_pc.squeeze().cpu().numpy()  # [128]

print(f"\n编码后的潜在向量形状: {latent_vector.shape}")
print(f"潜在向量值:\n{latent_vector}")



###

# 将重建的点云转换回 numpy
restoration_np = restoration.squeeze().permute(1, 0).cpu().numpy()  # [1024, 3]
# print(restoration_np)
# exit()
# 可视化原始点云和重建点云
print("\n正在可视化原始点云和重建点云...")

# 创建两个点云对象，使用不同颜色
# 原始点云：蓝色
original_pcd = o3d.geometry.PointCloud()
original_pcd.points = o3d.utility.Vector3dVector(points)  # 使用预处理后的点云
original_pcd.paint_uniform_color([0, 0, 1])  # 蓝色

# 重建点云：红色
restored_pcd = o3d.geometry.PointCloud()
restored_pcd.points = o3d.utility.Vector3dVector(restoration_np)
restored_pcd.paint_uniform_color([1, 0, 0])  # 红色

# 为了更好的可视化，将重建点云稍微偏移
# 这样可以避免两个点云完全重叠
offset = np.array([2.0, 0, 0])  # 在X方向偏移2个单位
restored_pcd.translate(offset)

# 创建坐标系
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

# 同时显示两个点云和坐标系
# o3d.visualization.draw_geometries(
#     [original_pcd, restored_pcd, coordinate_frame],
#     window_name="点云对比: 原始(蓝色) vs 重建(红色)",
#     width=1024,
#     height=768,
#     point_show_normal=False
# )

# # 可选：分别显示两个点云进行比较
# print("\n分别显示原始点云和重建点云...")

# # 显示原始点云
# original_pcd_no_offset = o3d.geometry.PointCloud()
# original_pcd_no_offset.points = o3d.utility.Vector3dVector(points)
# original_pcd_no_offset.paint_uniform_color([0, 0.5, 1])  # 浅蓝色
# o3d.visualization.draw_geometries(
#     [original_pcd_no_offset, coordinate_frame],
#     window_name="原始点云",
#     width=800,
#     height=600,
#     point_show_normal=False
# )

# # 显示重建点云
# restored_pcd_no_offset = o3d.geometry.PointCloud()
# restored_pcd_no_offset.points = o3d.utility.Vector3dVector(restoration_np)
# restored_pcd_no_offset.paint_uniform_color([1, 0.2, 0.2])  # 浅红色
# o3d.visualization.draw_geometries(
#     [restored_pcd_no_offset, coordinate_frame],
#     window_name="重建点云",
#     width=800,
#     height=600,
#     point_show_normal=False
# )

# 可选：计算重建误差
print("\n计算重建误差...")
mse = np.mean((points - restoration_np) ** 2)
print(f"均方误差 (MSE): {mse:.6f}")

# 可选：显示两个点云重叠在一起（使用不同颜色）
print("\n显示重叠的点云（原始蓝色，重建红色）...")
combined_pcd = o3d.geometry.PointCloud()
combined_pcd.points = o3d.utility.Vector3dVector(np.vstack([points, restoration_np]))

# 为组合点云创建颜色数组
colors = np.vstack([
    np.tile([0, 0, 1], (points.shape[0], 1)),  # 原始点：蓝色
    np.tile([1, 0, 0], (restoration_np.shape[0], 1))  # 重建点：红色
])
combined_pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries(
    [combined_pcd, coordinate_frame],
    window_name="重叠显示: 原始(蓝) + 重建(红)",
    width=1024,
    height=768,
    point_show_normal=False
)
