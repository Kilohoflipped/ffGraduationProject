import numpy as np
from tifffile import imread, imwrite
import cv2


class HoughTransform:
    def __init__(self, picPath):
        edge = imread(picPath)  # 二值图 (0 或 255) 得到 canny边缘检测的结果
        if not len(edge.shape) == 2:
            edge = edge[:, :, 1]
        # edge = edge[1:-1, 1:-1]
        lines = self.lines_detector_hough(edge)

        finalImg = self.draw_lines(edge, lines)
        finalImg = finalImg.astype(np.uint8)
        imwrite('testHoughed.tif', finalImg)

    def lines_detector_hough(self, edge, ThetaDim=None, DistStep=None, threshold=None, halfThetaWindowSize=2,
                             halfDistWindowSize=None):
        """
        :param edge: 经过边缘检测得到的二值图
        :param ThetaDim: hough空间中theta轴的刻度数量(将[0,pi)均分为多少份),反应theta轴的粒度,越大粒度越细
        :param DistStep: hough空间中dist轴的划分粒度,即dist轴的最小单位长度
        :param threshold: 投票表决认定存在直线的起始阈值
        :return: 返回检测出的所有直线的参数(dist,theta)
        """
        imgsize = edge.shape
        if ThetaDim is None:
            ThetaDim = 180
        if DistStep is None:
            DistStep = 1
        MaxDist = np.sqrt(imgsize[0] ** 2 + imgsize[1] ** 2)  # 计算最大距离
        DistDim = int(np.ceil(MaxDist / DistStep))  # 极径投票单元的个数

        if halfDistWindowSize is None:  # 计算阈值化用的距离半窗口大小
            halfDistWindowSize = int(DistDim / 50)

        Theta = np.linspace(-np.pi, np.pi, ThetaDim, endpoint=False)  # 通过参数初始化角度向量
        sinTheta = np.sin(Theta)  # 计算正弦向量
        cosTheta = np.cos(Theta)  # 计算余弦向量

        indexOfNotZero = np.flatnonzero(edge)  # 找到所有非0值的线性索引
        indexOfNotZeroX, indexOfNotZeroY = np.unravel_index(indexOfNotZero, edge.shape)  # 将线性索引转换为像素图的坐标索引
        xCosOfNotZero = np.outer(indexOfNotZeroX, cosTheta)  # 对于每个x计算在所有theta上的投票
        ySinOfNotZero = np.outer(indexOfNotZeroY, sinTheta)  # 对于每个y计算在所有theta上的投票

        rOfNotZero = (xCosOfNotZero + ySinOfNotZero) * DistDim / MaxDist  # 计算每个非零的坐标点对于每个theta上r的投票值
        rOfNotZero = np.floor(rOfNotZero).astype(int)  # 向下取整，得到具体的投票单元索引
        rOfNotZero = np.abs(rOfNotZero)  # 将距离取绝对值
        # theta的范围是[-pi,pi). 在这里将[pi,pi)进行了线性映射.类似的,也对Dist轴进行了线性映射
        accumulator = np.apply_along_axis(lambda x: np.bincount(x.astype(int), minlength=DistDim), 0, rOfNotZero)

        M = accumulator.max()
        if threshold is None:  # 如果没有给定阈值，计算阈值
            # threshold = np.mean(accumulator) + np.std(accumulator) * 1.5
            threshold = int(M * 2.3875 / 10)
        result = np.array(np.where(accumulator > threshold))  # 阈值化

        indexOfNonMaxR = result[0]
        indexOfNonMaxTheta = result[1]
        windowLeftIndices = np.maximum(indexOfNonMaxR - halfDistWindowSize + 1, 0)
        windowRightIndices = np.minimum(indexOfNonMaxR + halfDistWindowSize, accumulator.shape[0])
        windowTopIndices = np.maximum(indexOfNonMaxTheta - halfThetaWindowSize + 1, 0)
        windowBottomIndices = np.minimum(indexOfNonMaxTheta + halfThetaWindowSize, accumulator.shape[1])
        eightNeighborhood = [accumulator[windowLeftIndices[i]:windowRightIndices[i],
                             windowTopIndices[i]:windowBottomIndices[i]].max()
                             for i in range(windowBottomIndices.shape[0])]
        MaxMask = accumulator[result[0], result[1]] >= eightNeighborhood
        result = result[:, MaxMask]  # 非极大值抑制

        result = result.astype(np.float64)
        result[0] = result[0] * MaxDist / DistDim
        result[1] = result[1] * np.pi / ThetaDim

        return result

    def draw_lines(self, edge, lines, color=(0, 0, 255)):
        """
        :param edge: 原图的二值化边缘图像
        :param r: 直线参数r
        :param theta: 直线参数theta
        """
        r = lines[0]  # 极径
        theta = lines[1]  # 极角
        img = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)  # 通过cv2画图
        height, width = img.shape[:2]  # 获取图像的尺寸
        cosTheta = np.cos(theta)
        sinTheta = np.sin(theta)
        for i in range(len(r)):
            if sinTheta[i] != 0:  # 非竖直直线
                pltX1 = int(0)
                pltX2 = int(width)
                pltY1 = int((r[i] - pltX1 * cosTheta[i]) / sinTheta[i])
                pltY2 = int((r[i] - pltX2 * cosTheta[i]) / sinTheta[i])
            elif sinTheta[i] == 0:  # 竖直直线
                pltX1 = int(r[i] / cosTheta[i])
                pltX2 = int(r[i] / cosTheta[i])
                pltY1 = 0
                pltY2 = height
            cv2.line(img, (pltX1, pltY1), (pltX2, pltY2), color, 2)
        return img
