import math
import numpy as np
from tifffile import imread, imwrite


class CannyDetection:
    def __init__(self, picPath):
        sobelKernelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobelKernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        img = imread(picPath)
        sigma = 1.52  # 设置高斯滤波器的标准差，用来控制高斯核的大小
        dim = int(np.round(6 * sigma + 1))  # 通过3sigma原则计算高斯核纬度大小
        if dim % 2 == 0:  # 确保高斯核的大小为奇数，保证高斯核有一个中心点
            dim += 1

        linear_Gaussian_filter = [np.abs(t - (dim // 2)) for t in range(dim)]  # 计算高斯核中的每个点到中心的下标距离,
        # 计算高斯核中每个位置的函数值
        linear_Gaussian_filter = np.array(
            [[self.OneDimensionStandardNormalDistribution(t, sigma) for t in linear_Gaussian_filter]])
        linear_Gaussian_filter = linear_Gaussian_filter / linear_Gaussian_filter.sum()  # 归一化

        img2 = self._2DimDividedConvolve(linear_Gaussian_filter, img)  # 二位独立卷积，分别卷积
        img3 = self.convolve(sobelKernelX, img2, [1, 1, 1, 1], [1, 1])  # 横向solbel，计算横向梯度
        img4 = self.convolve(sobelKernelY, img2, [1, 1, 1, 1], [1, 1])  # 纵向solbel，计算纵向梯度

        gradiant_length = (img3 ** 2 + img4 ** 2) ** (1.0 / 2)  # 计算综合梯度模长

        img3 = img3.astype(np.float64)
        img4 = img4.astype(np.float64)
        img3[img3 == 0] = 1e-8  # 防止分母为0进而出现NaN
        gradiant_tangent = img4 / img3  # 计算梯度方向

        finalImg = self.DecideAndConnectEdge(gradiant_length, gradiant_tangent)
        finalImg[finalImg == 0] = 0
        finalImg[finalImg > 0] = 255
        finalImg = finalImg.astype(np.uint8)
        imwrite('testCannyed.tif', finalImg)

    def convolve(self, filter, imgMat, padding, strides):
        result = None
        filterSize = filter.shape  # 卷积核大小
        imgSize = imgMat.shape
        if len(filterSize) == 2:  # 如果是二维卷积核
            if len(imgSize) == 3:  # 如果图片有多个通道
                channel = []
                for i in range(imgSize[-1]):  # 对每个通道分别求值
                    # 对图片进行填充
                    paddedMat = np.pad(imgMat[:, :, i], ((padding[0], padding[1]), (padding[2], padding[3])),
                                       'constant')
                    temp = []
                    for j in range(0, imgSize[0], strides[1]):  # 使用双重循环遍历每一个像素点
                        temp.append([])
                        for k in range(0, imgSize[1], strides[0]):
                            val = (filter * paddedMat[j:j + filterSize[0], k:k + filterSize[1]]).sum()
                            temp[-1].append(val)
                    channel.append(np.array(temp))

                channel = tuple(channel)  # 元组
                result = np.dstack(channel)  # 堆叠
            elif len(imgSize) == 2:  # 如果图片只有一个通道
                channel = []
                paddedMat = np.pad(imgMat, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
                for j in range(0, imgSize[0], strides[1]):
                    channel.append([])  # 另起一行
                    for k in range(0, imgSize[1], strides[0]):
                        val = (filter * paddedMat[j:j + filterSize[0], k:k + filterSize[1]]).sum()
                        channel[-1].append(val)  # 行内末尾追加
                result = np.array(channel)
        return result

    def _linearConvolve(self, filter, mat, padding=None, strides=None):
        if strides is None:
            strides = [1, 1]
        result = None  # 初始化结果
        filterSize = filter.shape  # 卷积核大小
        if len(filterSize) == 2 and 1 in filterSize:  # 如果卷积核是2维且是线性的
            if padding is None or len(padding) < 2:  # 如果没有给出填充因子，则给出填充因子的大小
                if filterSize[1] == 1:  # 如果滤波器是纵的
                    padding = [filterSize[0] // 2, filterSize[0] // 2]  # 计算原图像上下需要补的行数
                elif filterSize[0] == 1:  # 如果滤波器是横的
                    padding = [filterSize[1] // 2, filterSize[1] // 2]  # 计算原图像左右需要补的行数
            if filterSize[0] == 1:  # 如果滤波器是横的
                result = self.convolve(filter, mat, [0, 0, padding[0], padding[1]], strides)
            elif filterSize[1] == 1:  # 如果滤波器是纵的
                result = self.convolve(filter, mat, [padding[0], padding[1], 0, 0], strides)
        return result

    def _2DimDividedConvolve(self, filter, mat):
        result = None  # 初始化结果
        if 1 in filter.shape:  # 如果卷积核是线性的，即Nx1
            result = self._linearConvolve(filter, mat)  # 卷积
            result = self._linearConvolve(filter.T, result)  # 转置再卷积
            # 等价于二维卷积
        return result

    def DecideAndConnectEdge(self, g_l, g_t, threshold=None):
        if threshold == None:
            lower_boundary = g_l.mean() * 0.5
            threshold = [lower_boundary, lower_boundary * 3]
        result = np.zeros(g_l.shape)

        for i in range(g_l.shape[0]):
            for j in range(g_l.shape[1]):
                isLocalExtreme = True
                eight_neiborhood = g_l[max(0, i - 1):min(i + 2, g_l.shape[0]), max(0, j - 1):min(j + 2, g_l.shape[1])]
                if eight_neiborhood.shape == (3, 3):
                    if g_t[i, j] <= -1:
                        x = 1 / g_t[i, j]
                        first = eight_neiborhood[0, 1] + (eight_neiborhood[0, 1] - eight_neiborhood[0, 0]) * x
                        x = -x
                        second = eight_neiborhood[2, 1] + (eight_neiborhood[2, 2] - eight_neiborhood[2, 1]) * x
                        if not (g_l[i, j] > first and g_l[i, j] > second):
                            isLocalExtreme = False
                    elif g_t[i, j] >= 1:
                        x = 1 / g_t[i, j]
                        first = eight_neiborhood[0, 1] + (eight_neiborhood[0, 2] - eight_neiborhood[0, 1]) * x
                        x = -x
                        second = eight_neiborhood[2, 1] + (eight_neiborhood[2, 1] - eight_neiborhood[2, 0]) * x
                        if not (g_l[i, j] > first and g_l[i, j] > second):
                            isLocalExtreme = False
                    elif g_t[i, j] >= 0 and g_t[i, j] < 1:
                        y = g_t[i, j]
                        first = eight_neiborhood[1, 2] + (eight_neiborhood[0, 2] - eight_neiborhood[1, 2]) * y
                        y = -y
                        second = eight_neiborhood[1, 0] + (eight_neiborhood[1, 0] - eight_neiborhood[2, 0]) * y
                        if not (g_l[i, j] > first and g_l[i, j] > second):
                            isLocalExtreme = False
                    elif g_t[i, j] < 0 and g_t[i, j] > -1:
                        y = g_t[i, j]
                        first = eight_neiborhood[1, 2] + (eight_neiborhood[1, 2] - eight_neiborhood[2, 2]) * y
                        y = -y
                        second = eight_neiborhood[1, 0] + (eight_neiborhood[0, 0] - eight_neiborhood[1, 0]) * y
                        if not (g_l[i, j] > first and g_l[i, j] > second):
                            isLocalExtreme = False
                if isLocalExtreme:
                    result[i, j] = g_l[i, j]  # 非极大值抑制

        result[result >= threshold[1]] = 255
        result[result <= threshold[0]] = 0

        result = self.judgeConnect(result, threshold)
        result[result != 255] = 0
        return result

    def judgeConnect(self, m2, threshold):
        e = 0.01
        s = []
        cood = []
        for i in range(m2.shape[0]):
            cood.append([])
            for j in range(m2.shape[1]):
                cood[-1].append([i, j])
                if abs(m2[i, j] - 255) < e:
                    s.append([i, j])
        cood = np.array(cood)

        while not len(s) == 0:
            index = s.pop()
            jud = m2[max(0, index[0] - 1):min(index[0] + 2, m2.shape[1]),
                  max(0, index[1] - 1):min(index[1] + 2, m2.shape[0])]
            jud_i = cood[max(0, index[0] - 1):min(index[0] + 2, cood.shape[1]),
                    max(0, index[1] - 1):min(index[1] + 2, cood.shape[0])]
            jud = (jud > threshold[0]) & (jud < threshold[1])
            jud_i = jud_i[jud]
            for i in range(jud_i.shape[0]):
                s.append(list(jud_i[i]))
                m2[jud_i[i][0], jud_i[i][1]] = 255

        return m2

    def OneDimensionStandardNormalDistribution(self, x, ssigma):  # 计算一维标准正态分布函数值的函数
        E = -0.5 / (ssigma * ssigma)
        return 1 / (math.sqrt(2 * math.pi) * ssigma) * math.exp(x * x * E)
