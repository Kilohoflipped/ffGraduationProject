import numpy as np
import laspy
from PIL import Image
from scipy import ndimage
from scipy.interpolate import griddata


class DSM:
    def __init__(self, laz_files_path, step_size):
        self.laz_file = None
        self.laz_files_path = laz_files_path
        self.step_size = step_size
        self.points_all = None
        self._read_laz_files()

        self.bounds = ((self.laz_file.header.x_min, self.laz_file.header.x_max),
                       (self.laz_file.header.y_min, self.laz_file.header.y_max))

        self.zBounds = (self.laz_file.header.z_min, self.laz_file.header.z_max)
        self.heightRaster = None
        self.heightRasterFilled = None
        self._rasterize()

        self.normalizedHeightRaster = np.zeros_like(self.heightRaster, dtype=float) + 255
        self.normalizedHeightRaster[self.heightRaster > 0] = (self.heightRaster[self.heightRaster > 0] -
                                                              np.min(self.heightRaster[self.heightRaster > 0])) / \
                                                             (np.max(self.heightRaster[self.heightRaster > 0]) -
                                                              np.min(self.heightRaster[self.heightRaster > 0]))

    def _read_laz_files(self):
        """
        从Laz文件中读取点云数据，并将其存储为Numpy数组
        """
        self.points_all = np.zeros((0, 3))
        self.laz_file = laspy.read(self.laz_files_path)
        self.points_all = np.append(self.points_all, self.laz_file.xyz, axis=0)

    def _rasterize(self):
        """
        对点云进行栅格化处理，得到高程的栅格图像
        """
        # 定义栅格的xy坐标
        (x_min, x_max), (y_min, y_max) = self.bounds
        xi = np.arange(x_min, x_max + self.step_size, self.step_size)  # 创建栅格，并且保证做有点都能囊括进栅格中
        yi = np.arange(y_min, y_max + self.step_size, self.step_size)  # 创建栅格，并且保证做有点都能囊括进栅格中
        XI, YI = np.meshgrid(xi, yi)

        # 计算每个点在栅格中的索引
        x_indices = ((self.points_all[:, 0] - x_min) / self.step_size).astype(int)  # 求出点云的栅格坐标
        y_indices = ((self.points_all[:, 1] - y_min) / self.step_size).astype(int)

        # 在栅格中聚合和插值
        self.heightRaster = np.zeros_like(XI, dtype=float)
        count = np.zeros_like(XI, dtype=int)
        np.add.at(self.heightRaster, (y_indices, x_indices), self.points_all[:, 2])  # 对应值叠加，将所有高程值叠加到栅格中
        np.add.at(count, (y_indices, x_indices), 1)  # 记录每个栅格内被叠加了多少个高程数据，用来计算平均值
        # 避免分母为0的情况
        self.heightRaster[count > 0] /= count[count > 0]  # 计算栅格高程平均值

    def fill_no_data(self):
        """
        对于在原始点云数据中没有赋予值的栅格，通过插值方法进行赋值
        """
        # 获取已知值和位置的坐标
        heightRasterSize = (self.heightRaster.shape[0], self.heightRaster.shape[1])
        known_coords = np.argwhere(self.heightRaster != 0)
        known_values = self.heightRaster[self.heightRaster != 0]

        # 生成整个矩阵的坐标网格
        yi, xi = np.mgrid[0:heightRasterSize[0], 0:heightRasterSize[1]]
        coords = np.vstack((yi.ravel(), xi.ravel())).T

        # 使用线性插值填充缺失值
        self.heightRasterFilled = griddata(known_coords, known_values, coords, method='linear').reshape(
            heightRasterSize)

        # 如果您还想要填充边缘的NaN值，可以使用其他方法，如“nearest”插值
        filled_matrix = griddata(known_coords, known_values, coords, method='nearest').reshape(heightRasterSize)
        self.heightRasterFilled[np.isnan(self.heightRasterFilled)] = filled_matrix[np.isnan(self.heightRasterFilled)]
        # 更新归一化的高程栅格图像
        self.normalizedHeightRaster[self.heightRasterFilled > 0] = (self.heightRasterFilled[
                                                                        self.heightRasterFilled > 0] -
                                                                    np.min(self.heightRasterFilled[
                                                                               self.heightRasterFilled > 0])) / \
                                                                   (np.max(self.heightRasterFilled[
                                                                               self.heightRasterFilled > 0]) -
                                                                    np.min(self.heightRasterFilled[
                                                                               self.heightRasterFilled > 0]))

    def save_dsm_image(self, file_path):
        """
        将高程栅格图像保存为灰度图像
        """
        dsm_image_array = (self.normalizedHeightRaster * 255).astype(np.uint8)
        dsm_image = Image.fromarray(dsm_image_array, mode='L')
        dsm_image.save(file_path)
