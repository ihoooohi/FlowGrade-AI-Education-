import cv2
import numpy as np
from skimage import measure
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass
class Shape:
    """表示流程图中的一个图形元素"""
    type: str  # 'rectangle', 'diamond', 'ellipse', 'parallelogram'
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]
    area: float
    text_region: Tuple[int, int, int, int] = None  # 文本区域的边界框

class ImageProcessor:
    """处理流程图图像，检测和分割图形元素"""
    
    def __init__(self):
        self.min_shape_area = 500  # 最小图形面积，过滤噪声
        self.shape_types = {
            'rectangle': self._is_rectangle,
            'diamond': self._is_diamond,
            'ellipse': self._is_ellipse,
            'parallelogram': self._is_parallelogram
        }
    
    def process(self, image_path: str) -> Tuple[np.ndarray, List[Shape]]:
        """
        处理流程图图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            处理后的图像和检测到的图形列表
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 膨胀边缘，连接断开的线条
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 处理和分类图形
        shapes = self._process_contours(contours, image.shape)
        
        # 创建可视化结果（用于调试）
        debug_image = image.copy()
        for shape in shapes:
            color = self._get_shape_color(shape.type)
            cv2.drawContours(debug_image, [shape.contour], 0, color, 2)
            x, y, w, h = shape.bbox
            cv2.putText(debug_image, shape.type, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image, shapes
    
    def _process_contours(self, contours: List[np.ndarray], image_shape: Tuple[int, int, int]) -> List[Shape]:
        """处理和分类检测到的轮廓"""
        shapes = []
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤小面积轮廓（噪声）
            if area < self.min_shape_area:
                continue
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算中心点
            center = (x + w // 2, y + h // 2)
            
            # 确定图形类型
            shape_type = self._determine_shape_type(contour, area)
            
            # 创建Shape对象
            shape = Shape(
                type=shape_type,
                contour=contour,
                bbox=(x, y, w, h),
                center=center,
                area=area
            )
            
            shapes.append(shape)
        
        return shapes
    
    def _determine_shape_type(self, contour: np.ndarray, area: float) -> str:
        """确定轮廓的图形类型"""
        for shape_type, check_func in self.shape_types.items():
            if check_func(contour, area):
                return shape_type
        
        # 默认为矩形
        return 'rectangle'
    
    def _is_rectangle(self, contour: np.ndarray, area: float) -> bool:
        """检查轮廓是否为矩形"""
        # 近似多边形
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 矩形应该有4个顶点
        if len(approx) == 4:
            # 计算边界矩形
            x, y, w, h = cv2.boundingRect(approx)
            rect_area = w * h
            
            # 计算轮廓与矩形的面积比
            area_ratio = area / rect_area
            
            # 矩形的面积比应接近1
            return area_ratio > 0.8
        
        return False
    
    def _is_diamond(self, contour: np.ndarray, area: float) -> bool:
        """检查轮廓是否为菱形"""
        # 近似多边形
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 菱形应该有4个顶点
        if len(approx) == 4:
            # 计算各边长度
            side_lengths = []
            for i in range(4):
                p1 = approx[i][0]
                p2 = approx[(i+1) % 4][0]
                side_lengths.append(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))
            
            # 计算对边长度比
            ratio1 = side_lengths[0] / side_lengths[2] if side_lengths[2] != 0 else 0
            ratio2 = side_lengths[1] / side_lengths[3] if side_lengths[3] != 0 else 0
            
            # 菱形的对边应该近似相等
            is_diamond = (0.8 < ratio1 < 1.2) and (0.8 < ratio2 < 1.2)
            
            # 检查是否为倾斜的矩形（菱形）
            if is_diamond:
                # 计算边界矩形
                x, y, w, h = cv2.boundingRect(approx)
                rect_area = w * h
                
                # 菱形的面积比应小于矩形
                area_ratio = area / rect_area
                return area_ratio < 0.8
        
        return False
    
    def _is_ellipse(self, contour: np.ndarray, area: float) -> bool:
        """检查轮廓是否为椭圆形"""
        if len(contour) < 5:  # 椭圆拟合需要至少5个点
            return False
        
        # 拟合椭圆
        try:
            ellipse = cv2.fitEllipse(contour)
            (center_x, center_y), (width, height), angle = ellipse
            
            # 计算椭圆面积
            ellipse_area = np.pi * (width/2) * (height/2)
            
            # 计算轮廓与椭圆的面积比
            area_ratio = area / ellipse_area
            
            # 椭圆的面积比应接近1
            return 0.85 < area_ratio < 1.15
        except:
            return False
    
    def _is_parallelogram(self, contour: np.ndarray, area: float) -> bool:
        """检查轮廓是否为平行四边形"""
        # 近似多边形
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 平行四边形应该有4个顶点
        if len(approx) == 4:
            # 计算各边的斜率
            slopes = []
            for i in range(4):
                p1 = approx[i][0]
                p2 = approx[(i+1) % 4][0]
                
                # 避免除以零
                if p2[0] - p1[0] == 0:
                    slope = float('inf')
                else:
                    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                
                slopes.append(slope)
            
            # 平行四边形的对边斜率应该相等
            is_parallel = (abs(slopes[0] - slopes[2]) < 0.1) and (abs(slopes[1] - slopes[3]) < 0.1)
            
            # 不是矩形或菱形
            if is_parallel and not self._is_rectangle(contour, area) and not self._is_diamond(contour, area):
                return True
        
        return False
    
    def _get_shape_color(self, shape_type: str) -> Tuple[int, int, int]:
        """根据图形类型返回颜色（用于可视化）"""
        colors = {
            'rectangle': (0, 255, 0),      # 绿色
            'diamond': (0, 0, 255),        # 红色
            'ellipse': (255, 0, 0),        # 蓝色
            'parallelogram': (255, 255, 0) # 青色
        }
        return colors.get(shape_type, (128, 128, 128))  # 默认灰色 