import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


def fibonacci_sphere(num_samples):
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    y = np.linspace(1, -1, num_samples)
    radius = np.sqrt(1 - y * y)
    theta = phi * np.arange(num_samples)
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.stack([x, y, z]).T


def edge_detection(points, n_neighbors=50, threhold=np.pi / 2):
    tree = KDTree(points)
    dists, indices = tree.query(points[:, :3], k=n_neighbors + 1)

    A = points[indices.flatten(), :3].reshape((n_points, n_neighbors + 1, 3))
    A = np.concatenate([A, np.ones((n_points, n_neighbors + 1, 1))], axis=2)
    ATA = A.swapaxes(1, 2) @ A

    eigenvalues, eigenvectors = np.linalg.eig(ATA)
    eigenvectors_max = eigenvalues.argmin(axis=1)
    c = eigenvectors[np.arange(n_points), :, eigenvectors_max].reshape((n_points, 1, 4))

    A_dash = A[:, :, :3] - np.sum(A * c, axis=2, keepdims=True) * c[:, :, :3]
    PQ = A_dash - A_dash[:, 0, :3].reshape((n_points, 1, 3))
    PQ = PQ[:, 1:, :] / (np.linalg.norm(PQ[:, 1:, :], axis=2, keepdims=True) + 1e-8)

    PQ_cos = np.sum(PQ[:, 0:1, :] * PQ[:, 1:, :], axis=2)
    PQ_sin_temp = np.cross(PQ[:, 0:1, :], PQ[:, 1:, :], axis=2)
    PQ_sign = np.sum(PQ_sin_temp * c[:, :, :3], axis=2)
    PQ_sin = np.linalg.norm(PQ_sin_temp, axis=2) * PQ_sign
    PQ_theta = np.arctan2(PQ_sin, PQ_cos)
    PQ_theta = np.where(PQ_theta > 0, PQ_theta, PQ_theta + 2 * np.pi)
    PQ_theta = np.sort(PQ_theta, axis=1)
    PQ_theta = np.concatenate([np.zeros((n_points, 1)), PQ_theta, 2 * np.pi * np.ones((n_points, 1))], axis=1)
    PQ_theta_delta = np.max(np.diff(PQ_theta, axis=1), axis=1) > threhold

    edge_index = PQ_theta_delta

    return edge_index


if __name__ == "__main__":
    n_points = 10000
    points = fibonacci_sphere(n_points)

    ray_dir_1 = np.array([1, 0, 0])
    ray_dir_2 = np.array([0, 1, 0])
    ray_dir_3 = np.array([0, 0, 1])

    points = points[np.arccos(np.dot(points, ray_dir_1)) > np.pi / 6]
    points = points[np.arccos(np.dot(points, ray_dir_2)) > np.pi / 6]
    points = points[np.arccos(np.dot(points, ray_dir_3)) > np.pi / 6]
    points = points[np.arccos(np.dot(points, -ray_dir_1)) > np.pi / 6]
    points = points[np.arccos(np.dot(points, -ray_dir_2)) > np.pi / 6]
    points = points[np.arccos(np.dot(points, -ray_dir_3)) > np.pi / 6]
    n_points = points.shape[0]

    edge_index = edge_detection(points)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points[~edge_index])
    pcd1.colors = o3d.utility.Vector3dVector(np.ones(((~edge_index).sum(), 3))*0.75)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points[edge_index])

    labels = np.array(pcd2.cluster_dbscan(eps=0.1, min_points=5, print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd2.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pcd1, pcd2])
