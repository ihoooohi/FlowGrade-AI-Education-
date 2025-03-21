import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from paddleocr import PaddleOCR

class TextRegion:
    """表示图像中的文本区域"""
    def __init__(self, text: str, bbox: Tuple[int, int, int, int], confidence: float):
        """
        初始化文本区域
        
        Args:
            text: 识别的文本内容
            bbox: 文本的边界框 (x, y, w, h)
            confidence: OCR识别的置信度
        """
        self.text = text
        self.bbox = bbox  # (x, y, w, h)
        self.confidence = confidence
        self.center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        self.associated_shape = None  # 关联的图形元素
    
    def __repr__(self):
        return f"TextRegion(text='{self.text}', bbox={self.bbox}, confidence={self.confidence:.2f})"

class OCREngine:
    """使用PaddleOCR进行文本识别的引擎"""
    
    def __init__(self, lang='ch'):
        """
        初始化OCR引擎
        
        Args:
            lang: 语言代码，默认为中文
        """
        # 使用静默模式初始化PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        self.min_confidence = 0.5  # 最小置信度阈值
    
    def extract_text(self, image: np.ndarray) -> List[TextRegion]:
        """
        从图像中提取文本
        
        Args:
            image: 输入图像
            
        Returns:
            文本区域列表
        """
        # 确保图像是BGR格式
        if len(image.shape) == 2:  # 灰度图
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 运行OCR
        result = self.ocr.ocr(image, cls=True)
        
        # 处理结果
        text_regions = []
        
        if result is None or len(result) == 0:
            return text_regions
        
        # PaddleOCR返回格式: [[[x1,y1],[x2,y1],[x2,y2],[x1,y2]], (text, confidence)]
        for line in result[0]:
            if line is None or len(line) != 2:
                continue
            
            points, (text, confidence) = line
            
            # 过滤低置信度结果
            if confidence < self.min_confidence:
                continue
            
            # 计算边界框
            points = np.array(points)
            x_min = int(min(points[:, 0]))
            y_min = int(min(points[:, 1]))
            x_max = int(max(points[:, 0]))
            y_max = int(max(points[:, 1]))
            
            width = x_max - x_min
            height = y_max - y_min
            
            # 创建TextRegion对象
            text_region = TextRegion(
                text=text,
                bbox=(x_min, y_min, width, height),
                confidence=confidence
            )
            
            text_regions.append(text_region)
        
        return text_regions
    
    def visualize_text_regions(self, image: np.ndarray, text_regions: List[TextRegion]) -> np.ndarray:
        """
        可视化文本区域（用于调试）
        
        Args:
            image: 输入图像
            text_regions: 文本区域列表
            
        Returns:
            带有文本区域标注的图像
        """
        # 创建副本
        vis_image = image.copy()
        
        for region in text_regions:
            x, y, w, h = region.bbox
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制文本
            cv2.putText(vis_image, f"{region.text} ({region.confidence:.2f})", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_image 