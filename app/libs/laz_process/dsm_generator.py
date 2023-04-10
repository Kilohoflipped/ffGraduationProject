import numpy as np
import laspy
import os
from PIL import Image
from scipy import ndimage
from scipy.interpolate import griddata

class DSM:
    def __init__(self, laz_files_dir, step_size):
        self.laz_file = None
        self.laz_files_dir = laz_files_dir
        self.step_size = step_size
        self.points_all = self._read_laz_files()
        self.bounds = ((np.min(self.points_all[:, 0]), np.max(self.points_all[:, 0])),
                       (np.min(self.points_all[:, 1]), np.max(self.points_all[:, 1])))

        self.zBounds = (np.min(self.points_all[:, 2]), np.max(self.points_all[:, 2]))

        self.dsm = self._interpolate()
        self.normalized_dsm = (self.dsm - np.min(self.dsm)) / (np.max(self.dsm) - np.min(self.dsm))

    def _read_laz_files(self):
        """
        从Laz文件中读取点云数据，并将其存储为Numpy数组
        """
        points_all = np.zeros((0, 3))
        for file in os.listdir(self.laz_files_dir):
            if file.endswith('.laz'):
                file_path = os.path.join(self.laz_files_dir, file)
                self.laz_file = laspy.read(file_path)
                points = self.laz_file.xyz
                points_all = np.append(points_all, points, axis=0)
        return points_all

    def _interpolate(self):
        """
        对点云进行插值处理，得到高程的栅格图像
        """
        # 定义栅格的xy坐标
        (x_min, x_max), (y_min, y_max) = self.bounds
        xi = np.arange(x_min, x_max + self.step_size, self.step_size)
        yi = np.arange(y_min, y_max + self.step_size, self.step_size)
        XI, YI = np.meshgrid(xi, yi)

        # 计算每个点在栅格中的索引
        x_indices = ((self.points_all[:, 0] - x_min) / self.step_size).astype(int)
        y_indices = ((self.points_all[:, 1] - y_min) / self.step_size).astype(int)

        # 在栅格中聚合和插值
        raster = np.zeros_like(XI, dtype=float)
        count = np.zeros_like(XI, dtype=int)
        np.add.at(raster, (y_indices, x_indices), self.points_all[:, 2])
        np.add.at(count, (y_indices, x_indices), 1)
        # 避免分母为0的情况
        raster[count > 0] /= count[count > 0]

        return raster

    def fill_no_data(self):
        """
        对于在原始点云数据中没有赋予值的栅格，通过插值方法进行赋值
        """
        # 找到没有值的栅格
        no_data_indices = np.isclose(self.dsm, 0, atol=0, rtol=1e-8)

        # 获取已知值的栅格和对应的坐标
        known_values = self.dsm[~no_data_indices]
        (x_min, x_max), (y_min, y_max) = self.bounds
        x_coords, y_coords = np.meshgrid(np.arange(x_min, x_max + self.step_size, self.step_size),
                                         np.arange(y_min, y_max + self.step_size, self.step_size))
        known_coords = np.column_stack((x_coords[~no_data_indices].flatten(),
                                        y_coords[~no_data_indices].flatten()))

        # 获取未知值的栅格坐标
        unknown_coords = np.column_stack((x_coords[no_data_indices].flatten(),
                                          y_coords[no_data_indices].flatten()))

        # 使用插值方法为未知值的栅格赋值
        interpolated_values = griddata(known_coords, known_values, unknown_coords, method='cubic')

        # 在插值后检查 NaN 值
        nan_indices = np.isnan(interpolated_values)

        # 如果存在 NaN 值，使用最近邻插值方法处理它们
        if np.any(nan_indices):
            interpolated_values[nan_indices] = griddata(known_coords, known_values, unknown_coords[nan_indices],
                                                        method='nearest')

        # 更新DSM中未知值的栅格
        self.dsm[no_data_indices] = np.nan_to_num(interpolated_values, nan=np.min(self.dsm))

        # 更新归一化的高程栅格图像
        self.normalized_dsm = (self.dsm - np.min(self.dsm)) / (np.max(self.dsm) - np.min(self.dsm))

    def save_dsm_image(self, file_path):
        """
        将高程栅格图像保存为灰度图像
        """
        dsm_image_array = (self.normalized_dsm * 255).astype(np.uint8)
        dsm_image = Image.fromarray(dsm_image_array, mode='L')
        dsm_image.save(file_path)