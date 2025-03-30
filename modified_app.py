import os
import base64
import json
import uuid
import re
import time
import requests
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# 加载环境变量
load_dotenv(override=True)

# 尝试导入OpenCV
try:
    import cv2
    import numpy as np
    CV_AVAILABLE = True
    print("OpenCV 已成功导入")
except ImportError as e:
    print(f"警告: OpenCV 未安装，将使用模拟图像处理功能。错误: {str(e)}")
    CV_AVAILABLE = False

# 尝试导入PaddleOCR
try:
    # 尝试导入所需的依赖项
    missing_deps = []
    try:
        import shapely
        print("shapely 已成功导入")
    except ImportError as e:
        missing_deps.append(f"shapely: {str(e)}")
    
    try:
        import pyclipper
        print("pyclipper 已成功导入")
    except ImportError as e:
        missing_deps.append(f"pyclipper: {str(e)}")
    
    try:
        import albumentations
        print("albumentations 已成功导入")
    except ImportError as e:
        missing_deps.append(f"albumentations: {str(e)}")
    
    try:
        import albucore
        print("albucore 已成功导入")
    except ImportError as e:
        missing_deps.append(f"albucore: {str(e)}")
    
    try:
        import fire
        print("fire 已成功导入")
    except ImportError as e:
        missing_deps.append(f"fire: {str(e)}")
    
    try:
        import lmdb
        print("lmdb 已成功导入")
    except ImportError as e:
        missing_deps.append(f"lmdb: {str(e)}")
    
    # 如果有缺失的依赖项，则抛出异常
    if missing_deps:
        raise ImportError(f"缺少依赖项: {', '.join(missing_deps)}")
    
    # 尝试导入PaddleOCR
    try:
        from paddleocr import PaddleOCR
        print("PaddleOCR 模块已成功导入")
        
        # 尝试初始化PaddleOCR以验证是否可用
        try:
            ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
            PADDLE_OCR_AVAILABLE = True
            print("PaddleOCR 已成功初始化")
        except Exception as e:
            print(f"警告: PaddleOCR 导入成功但初始化失败，将使用模拟OCR功能。错误: {str(e)}")
            PADDLE_OCR_AVAILABLE = False
    except ImportError as e:
        print(f"警告: PaddleOCR 模块导入失败，将使用模拟OCR功能。错误: {str(e)}")
        PADDLE_OCR_AVAILABLE = False
except ImportError as e:
    print(f"警告: PaddleOCR 依赖项导入失败，将使用模拟OCR功能。错误: {str(e)}")
    PADDLE_OCR_AVAILABLE = False

# 尝试导入OpenAI
try:
    import openai
    # 验证API密钥是否有效
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    print(f"API密钥: {api_key[:5]}...{api_key[-5:] if api_key and len(api_key) > 10 else 'None'}")
    if not api_key or api_key == "your_api_key_here":
        print("警告: DASHSCOPE_API_KEY 环境变量未设置或无效，将使用模拟AI评估功能")
        OPENAI_AVAILABLE = False
    else:
        OPENAI_AVAILABLE = True
        print("OpenAI 已成功导入")
except ImportError as e:
    print(f"警告: OpenAI 未安装，将使用模拟AI评估功能。错误: {str(e)}")
    OPENAI_AVAILABLE = False

# 定义千问视觉OCR是否可用
QWEN_OCR_AVAILABLE = OPENAI_AVAILABLE

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class MockImageProcessor:
    """模拟图像处理器"""
    def process(self, image_path):
        print(f"模拟处理图像: {image_path}")
        # 返回模拟的图像和图形列表
        return None, [
            {"type": "rectangle", "position": {"x": 100, "y": 100}, "center": (100, 100), "bbox": (80, 80, 40, 40), "text": "教师讲解", "contour": []},
            {"type": "diamond", "position": {"x": 200, "y": 200}, "center": (200, 200), "bbox": (180, 180, 40, 40), "text": "是否理解?", "contour": []},
            {"type": "ellipse", "position": {"x": 50, "y": 50}, "center": (50, 50), "bbox": (30, 30, 40, 40), "text": "开始", "contour": []},
            {"type": "parallelogram", "position": {"x": 300, "y": 300}, "center": (300, 300), "bbox": (280, 280, 40, 40), "text": "学生提交作业", "contour": []}
        ]

class MockOCREngine:
    """模拟OCR引擎"""
    def extract_text(self, image):
        print("模拟OCR文本提取")
        # 返回模拟的文本区域列表
        return [
            {"text": "教师讲解", "position": {"x": 100, "y": 100}, "center": (100, 100), "bbox": (80, 80, 40, 40), "confidence": 0.9},
            {"text": "是否理解?", "position": {"x": 200, "y": 200}, "center": (200, 200), "bbox": (180, 180, 40, 40), "confidence": 0.9},
            {"text": "开始", "position": {"x": 50, "y": 50}, "center": (50, 50), "bbox": (30, 30, 40, 40), "confidence": 0.9},
            {"text": "学生提交作业", "position": {"x": 300, "y": 300}, "center": (300, 300), "bbox": (280, 280, 40, 40), "confidence": 0.9}
        ]

    """模拟流程图分析器"""
    def analyze(self, shapes, text_regions):
        print("模拟流程图分析")
        # 返回模拟的结构化数据
        return {
            "nodes": [
                {"id": 0, "type": "start", "text": "开始", "position": {"x": 50, "y": 50}, "connections": [1]},
                {"id": 1, "type": "activity", "text": "教师讲解", "position": {"x": 100, "y": 100}, "connections": [2]},
                {"id": 2, "type": "decision", "text": "是否理解?", "position": {"x": 200, "y": 200}, "connections": [3, 1]},
                {"id": 3, "type": "input_output", "text": "学生提交作业", "position": {"x": 300, "y": 300}, "connections": [4]},
                {"id": 4, "type": "end", "text": "结束", "position": {"x": 350, "y": 350}, "connections": []}
            ],
            "connections": [
                {"from": 0, "to": 1, "label": ""},
                {"from": 1, "to": 2, "label": ""},
                {"from": 2, "to": 3, "label": "是"},
                {"from": 2, "to": 1, "label": "否"},
                {"from": 3, "to": 4, "label": ""}
            ],
            "statistics": {
                "total_nodes": 5,
                "node_types": {
                    "activity": 1,
                    "decision": 1,
                    "start": 1,
                    "end": 1,
                    "input_output": 1
                }
            }
        }

class RealFlowchartAnalyzer:
    """实际的流程图分析器"""
    def __init__(self):
        self.max_distance = 100  # 增加图形和文本匹配的最大距离
        self.connection_threshold = 200  # 增加连接判断的距离阈值
    
    def analyze(self, shapes, text_regions):
        """分析流程图，关联图形和文本"""
        print("开始分析流程图...")
        print(f"图形数量: {len(shapes)}")
        print(f"文本区域数量: {len(text_regions)}")
        
        # 1. 将文本关联到图形
        nodes = self._associate_text_with_shapes(shapes, text_regions)
        
        # 2. 标识节点类型
        self._identify_node_types(nodes)
        
        # 3. 识别连接关系
        connections = self._identify_connections(nodes)
        
        # 4. 计算统计信息
        statistics = self._calculate_statistics(nodes)
        
        # 检查并确保所有节点都有必要的字段
        for i, node in enumerate(nodes):
            if 'shape_type' not in node:
                print(f"警告: 节点 {i+1} 缺少shape_type字段，添加默认值'unknown'")
                node['shape_type'] = 'unknown'
            if 'text' not in node:
                print(f"警告: 节点 {i+1} 缺少text字段，添加默认值''")
                node['text'] = ''
        
        print(f"流程图分析完成，共 {len(nodes)} 个节点")
        
        return {
            "nodes": nodes,
            "connections": connections,
            "statistics": statistics
        }
    
    def _associate_text_with_shapes(self, shapes, text_regions):
        """将文本关联到图形"""
        if not text_regions:
            print("没有文本区域，跳过关联")
            
            # 创建没有文本的节点
            nodes = []
            for i, shape in enumerate(shapes):
                node = {
                    "id": i,
                    "shape_type": shape["type"],
                    "text": "",
                    "position": shape["position"] if "position" in shape else {"x": 0, "y": 0},
                    "bbox": shape["bbox"] if "bbox" in shape else {"x": 0, "y": 0, "width": 0, "height": 0},
                }
                nodes.append(node)
            
            return nodes
            
        # 按创建顺序依次关联
        text_index = 0
        total_texts = len(text_regions)
        
        print(f"开始按顺序关联，共有 {len(shapes)} 个图形和 {total_texts} 个文本区域")
        
        # 创建节点列表
        nodes = []
        
        for i, shape in enumerate(shapes):
            # 默认文本为空
            text = ""
            
            # 如果还有可用的文本区域，关联文本
            if text_index < total_texts:
                # 获取当前文本区域
                text_region = text_regions[text_index]
                text = text_region["text"]
                print(f"图形 {i} ({shape['type']}): 关联文本 '{text}'")
                # 移动到下一个文本区域
                text_index += 1
            else:
                print(f"图形 {i} ({shape['type']}): 文本已用完，不关联")
            
            # 创建节点
            node = {
                "id": i,
                "shape_type": shape["type"],
                "text": text,
                "position": shape["position"] if "position" in shape else {"x": 0, "y": 0},
                "bbox": shape["bbox"] if "bbox" in shape else {"x": 0, "y": 0, "width": 0, "height": 0},
            }
            
            nodes.append(node)
        
        return nodes
    
    def _identify_node_types(self, nodes):
        """识别节点类型并更新节点信息"""
        for i, node in enumerate(nodes):
            # 确保节点包含所有必要的字段
            if "shape_type" not in node:
                print(f"警告: 节点 {i} 缺少shape_type字段，跳过")
                continue
                
            # 确保节点有text字段
            if "text" not in node:
                print(f"警告: 节点 {i} 缺少text字段，设为空")
                node["text"] = ""
                
            # 根据图形类型确定节点类型
            node_type = self._map_shape_to_node_type(node["shape_type"], node["text"])
            
            # 更新节点类型
            node["type"] = node_type
            
            # 确保节点有connections字段
            if "connections" not in node:
                node["connections"] = []
                
            # print(f"标识节点: ID {i}, 类型 {node_type}, 文本 '{node['text']}'")
        
        return nodes
    
    def _map_shape_to_node_type(self, shape_type, text):
        """将图形类型映射到节点类型"""
        text_lower = text.lower() if text else ""
        
        if shape_type == "ellipse":
            if any(keyword in text_lower for keyword in ["开始", "起点", "start"]):
                return "start"
            elif any(keyword in text_lower for keyword in ["结束", "终点", "end"]):
                return "end"
            else:
                return "activity"  # 默认为活动
        
        elif shape_type == "菱形":
            return "决策"
        
        elif shape_type == "平行四边形":
            return "活动"
        
        elif shape_type == "矩形":
            return "活动"
            
        else:  # 其他形状
            print(f"未知图形类型: {shape_type}，默认为活动")
            return "活动"
    
    def _identify_connections(self, nodes):
        """识别节点之间的连接关系"""
        connections = []
        
        # 如果节点数量太少，直接返回空连接
        if len(nodes) <= 1:
            return connections
            
        # 对每个节点，找到最近的其他节点作为可能的连接
        for i, node in enumerate(nodes):
            node_center = (node["position"]["x"], node["position"]["y"])
            
            # 查找可能的连接
            possible_connections = []
            for j, other_node in enumerate(nodes):
                if i == j:  # 跳过自身
                    continue
                
                other_center = (other_node["position"]["x"], other_node["position"]["y"])
                
                # 计算距离
                distance = self._calculate_distance(node_center, other_center)
                
                # 如果距离小于阈值，添加到可能的连接
                if distance < self.connection_threshold:
                    possible_connections.append((j, distance))
            
            # 按距离排序
            possible_connections.sort(key=lambda x: x[1])
            
            # 取最近的几个节点作为连接
            max_connections = min(3, len(possible_connections))  # 最多连接数，但不超过可能的连接数
            for j, distance in possible_connections[:max_connections]:
                # 添加到节点的连接列表
                if j not in node["connections"]:
                    node["connections"].append(j)
                    
                    # 创建连接对象
                    connection = {
                        "from": i,
                        "to": j,
                        "label": ""  # 暂时不支持连接标签
                    }
                    
                    connections.append(connection)
                    # print(f"识别连接: 从节点 {i} 到节点 {j}, 距离 {distance:.2f}")
        
        return connections
    
    def _calculate_statistics(self, nodes):
        """计算流程图统计信息"""
        # 统计各类型节点数量
        node_types = {}
        for node in nodes:
            node_type = node["type"]
            if node_type in node_types:
                node_types[node_type] += 1
            else:
                node_types[node_type] = 1
        
        # 构建统计信息
        statistics = {
            "total_nodes": len(nodes),
            "node_types": node_types
        }
        
        return statistics
    
    def _calculate_distance(self, point1, point2):
        """计算两点之间的欧几里得距离"""
        try:
            return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
        except (TypeError, IndexError) as e:
            print(f"计算距离时出错: {e}, point1={point1}, point2={point2}")
            return float('inf')  # 返回无穷大表示无法计算距离

# class MockAIEvaluator:
#     """模拟AI评估器"""
#     def evaluate(self, structured_data, image_path):
#         print(f"模拟AI评估: {image_path}")
#         # 返回模拟的评估结果
#         return {
#             "scores": {
#                 "逻辑性": 8,
#                 "合理性": 7,
#                 "清晰度": 9,
#                 "创新性": 6,
#                 "总分": 75
#             },
#             "feedback": "这是一个示例反馈。在实际系统中，这里会显示基于AI分析的详细反馈。",
#             "符合规范": True,
#             "改进建议": [
#                 "建议1：可以考虑添加更多的决策点，使流程更加灵活。",
#                 "建议2：教师活动和学生活动的区分可以更加明确。",
#                 "建议3：可以考虑添加时间标注，以便更好地控制教学节奏。"
#             ],
#             "statistics": structured_data.get("statistics", {})
#         }

class RealImageProcessor:
    """实际的图像处理器"""
    def __init__(self):
        self.min_shape_area = 10000  # 进一步降低最小图形面积阈值，以便捕获更多的图形
        self.max_shape_area = 60000  # 最大图形面积阈值，过滤掉整个图像的轮廓
        
    def process(self, image_path):
        """处理流程图图像"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 保存原始图像尺寸
        original_height, original_width = image.shape[:2]
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用自适应阈值处理，更好地处理不同光照条件下的图像
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 形态学操作：开运算，去除小噪点
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 形态学操作：闭运算，连接断开的线条
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 创建调试图像
        debug_image = image.copy()
        
        # 保存预处理图像
        preprocess_path = os.path.join(app.config['UPLOAD_FOLDER'], f"preprocess_{uuid.uuid4()}.jpg")
        cv2.imwrite(preprocess_path, closed)
        print(f"预处理图像已保存: {preprocess_path}")
        
        # 查找轮廓 - 使用RETR_EXTERNAL只获取外部轮廓
        contours_external, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 查找轮廓 - 使用RETR_CCOMP获取层次结构
        contours_all, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # 合并两种轮廓检测结果 - 修复：确保所有轮廓都是相同类型
        contours = []
        for c in contours_external:
            if c.shape[0] >= 4:  # 至少需要4个点
                contours.append(c)
        for c in contours_all:
            if c.shape[0] >= 4 and not any(np.array_equal(c, existing) for existing in contours):  # 避免重复
                contours.append(c)
        
        # 处理和分类图形
        shapes = []
        
        # 过滤和处理轮廓
        for i, contour in enumerate(contours):
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤小面积和大面积轮廓
            if area < self.min_shape_area or area > self.max_shape_area:
                continue
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            
            # 计算面积比
            area_ratio = area / rect_area if rect_area > 0 else 0
            
            # 计算最小外接矩形面积比
            min_rect = cv2.minAreaRect(contour)
            min_rect_width, min_rect_height = min_rect[1]
            min_rect_area = min_rect_width * min_rect_height
        
            # 计算轮廓与最小外接矩形的面积比
            min_rect_area_ratio = area / min_rect_area if min_rect_area > 0 else 0
            
            # 过滤掉接近图像边缘的轮廓（可能是整个图像的边框）
            if x <= 5 or y <= 5 or x + w >= original_width - 5 or y + h >= original_height - 5:
                continue
            
            # 计算中心点
            center = (x + w // 2, y + h // 2)
            
            # 确定图形类型
            shape_type, confidence = self._determine_shape_type(contour, area)
            
            # 打印图形信息
            print(f"图形 {i} - 类型: {shape_type}, area_ratio: {area_ratio:.4f}, min_rect_area_ratio: {min_rect_area_ratio:.4f}")
            
            # 创建Shape对象
            shape = {
                "type": shape_type,
                "contour": contour.tolist(),
                "bbox": (x, y, w, h),
                "center": center,
                "area": area,
                "confidence": confidence,
                "position": {"x": center[0], "y": center[1]},
                "text": ""  # 初始化文本字段
            }
            
            # 在调试图像上绘制轮廓和类型
            color = (0, 255, 0)  # 默认绿色
            if shape_type == "diamond":
                color = (0, 0, 255)  # 红色
            elif shape_type == "ellipse":
                color = (255, 0, 0)  # 蓝色
            elif shape_type == "parallelogram":
                color = (0, 255, 255)  # 黄色
                
            cv2.drawContours(debug_image, [contour], 0, color, 2)
            cv2.putText(debug_image, f"{i}: {shape_type}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            shapes.append(shape)
        
        # 按照y坐标从上到下排序图形
        shapes.sort(key=lambda shape: shape["position"]["y"])
        
        # 重新编号并更新调试图像
        debug_image = image.copy()  # 重新创建调试图像
        for i, shape in enumerate(shapes):
            # 获取边界框
            x, y, w, h = shape["bbox"]
            
            # 重新绘制轮廓和编号
            contour = np.array(shape["contour"], dtype=np.int32)
            
            # 设置颜色
            color = (0, 255, 0)  # 默认绿色
            if shape["type"] == "diamond":
                color = (0, 0, 255)  # 红色
            elif shape["type"] == "ellipse":
                color = (255, 0, 0)  # 蓝色
            elif shape["type"] == "parallelogram":
                color = (0, 255, 255)  # 黄色
                
            cv2.drawContours(debug_image, [contour], 0, color, 2)
            cv2.putText(debug_image, f"{i}: {shape['type']}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 打印排序后的图形信息
            print(f"排序后 图形 {i} - 类型: {shape['type']}, 位置: y={shape['position']['y']}")
        
        # 保存调试图像
        debug_path = os.path.join(app.config['UPLOAD_FOLDER'], f"debug_{uuid.uuid4()}.jpg")
        cv2.imwrite(debug_path, debug_image)
        print(f"调试图像已保存: {debug_path}")
        
        # 打印识别到的图形数量
        print(f"识别到 {len(shapes)} 个图形，已按从上到下排序")
        
        return image, shapes
    
    def _determine_shape_type(self, contour, area):
        """确定轮廓的图形类型：矩形、平行四边形、菱形、圆角矩形"""
        # 近似多边形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 获取边界框和最小外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        
        # 计算轮廓与边界框的面积比
        area_ratio = area / rect_area
        
        # 计算宽高比
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # 获取最小外接矩形（可能有旋转）
        min_rect = cv2.minAreaRect(contour)
        min_rect_width, min_rect_height = min_rect[1]
        min_rect_area = min_rect_width * min_rect_height
        
        # 计算轮廓与最小外接矩形的面积比
        min_rect_area_ratio = area / min_rect_area if min_rect_area > 0 else 0
        
        # 获取轮廓的角点
        corners = self._get_corner_points(contour)
        
        
        
        # 非圆角矩形，但点数接近4的可能是四边形
        if 4 <= len(approx) <= 6:
            # 计算角点间的角度分布
            angles = self._calculate_corner_angles(corners)
            angle_std = np.std(angles) if angles else 0
            
            # 判断是否为菱形
            # 菱形特征：四个角点，对角线垂直，四条边长度相近
            if len(corners) == 4:
                # 检查对角线是否大致垂直
                diag1 = np.array(corners[2]) - np.array(corners[0])
                diag2 = np.array(corners[3]) - np.array(corners[1])
                dot_product = np.dot(diag1, diag2)
                diag_angle = abs(dot_product) / (np.linalg.norm(diag1) * np.linalg.norm(diag2)) if np.linalg.norm(diag1) * np.linalg.norm(diag2) > 0 else 1
                
                # 计算四条边的长度
                side_lengths = []
                for i in range(4):
                    next_i = (i + 1) % 4
                    side = np.linalg.norm(np.array(corners[next_i]) - np.array(corners[i]))
                    side_lengths.append(side)
                
                # 计算边长的标准差比例
                side_mean = np.mean(side_lengths)
                side_std = np.std(side_lengths)
                side_std_ratio = side_std / side_mean if side_mean > 0 else 1
                
                # 菱形条件：对角线接近垂直且四边长度接近相等
                if diag_angle < 0.3 and side_std_ratio < 0.3:
                    return "菱形", 0.9
            
            # 判断是否为平行四边形
            # 平行四边形特征：四个角点，对边平行，对边长度相近，但形状有倾斜
            # if len(corners) == 4:
            #     # 检查对边是否平行
            #     opposite_sides_parallel = True
            #     for i in range(2):
            #         side1 = np.array(corners[(i+1)%4]) - np.array(corners[i])
            #         side2 = np.array(corners[(i+3)%4]) - np.array(corners[(i+2)%4])
            #         side1_norm = side1 / np.linalg.norm(side1) if np.linalg.norm(side1) > 0 else np.array([0, 0])
            #         side2_norm = side2 / np.linalg.norm(side2) if np.linalg.norm(side2) > 0 else np.array([0, 0])
            #         dot = abs(np.dot(side1_norm, side2_norm))
            #         if dot < 0.9:  # 对边不够平行
            #             opposite_sides_parallel = False
            #             break
                
            #     # 检查是否有明显倾斜（区别于矩形）
            #     is_skewed = False
            #     for angle in angles:
            #         if abs(angle - 90) > 10:  # 角度偏离90度超过10度
            #             is_skewed = True
            #             break
                
            #     # 计算对边长度差异
            #     opposite_sides_similar = True
            #     for i in range(2):
            #         side1_len = np.linalg.norm(np.array(corners[(i+1)%4]) - np.array(corners[i]))
            #         side2_len = np.linalg.norm(np.array(corners[(i+3)%4]) - np.array(corners[(i+2)%4]))
            #         if abs(side1_len - side2_len) / max(side1_len, side2_len) > 0.15:  # 对边长度差异超过15%
            #             opposite_sides_similar = False
            #             break
                
            #     # 平行四边形条件：对边平行，对边长度相近，有明显倾斜
            #     if opposite_sides_parallel and opposite_sides_similar and is_skewed:
            #         # 附加检查：如果外接矩形面积比很高但明显有倾斜，那么是平行四边形
            #         if min_rect_area_ratio > 0.75 and is_skewed:
            #             return "parallelogram", 0.9
            #         # 一般情况下的平行四边形
            #         elif 0.7 < min_rect_area_ratio < 0.85:
            #             return "parallelogram", 0.85
            
            # 判断是否为矩形
            # 矩形特征：四个角点，四个角接近90度，面积比接近1
            if len(corners) == 4:
                # 检查角是否接近90度
                is_right_angled = True
                for angle in angles:
                    if abs(angle - 90) > 15:  # 允许15度的误差
                        is_right_angled = False
                        break
                
                # 矩形条件：各角接近90度，面积比高
                if is_right_angled and min_rect_area_ratio > 0.97:
                    return "矩形", 0.95
        
        # 检测是否为圆角矩形
        is_rounded_rect = self._is_rounded_rectangle(contour, approx, area)
        if is_rounded_rect and (area_ratio > 0.96 or area_ratio < 0.93):
            return "圆角矩形", 0.9
        
        # 默认情况，根据面积比判断
        if area_ratio > 0.97:
            return "矩形", 0.7
        elif 0.95 < area_ratio < 0.96:
            return "平行四边形", 0.7
        elif 0.5 < area_ratio < 0.75:
            return "菱形", 0.7
        else:
            return "未知", 0.5
    
    def _get_corner_points(self, contour):
        """获取轮廓的角点"""
        # 使用Shi-Tomasi角点检测算法获取角点
        if len(contour) < 4:
            return []
        
        # 创建掩膜图像
        mask = np.zeros((800, 800), dtype=np.uint8)  # 假设图像大小足够
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        # 使用Harris角点检测
        corners = cv2.goodFeaturesToTrack(mask, 8, 0.1, 10)
        
        if corners is None:
            return []
        
        # 将角点坐标转换为列表
        corners = [tuple(map(int, corner[0])) for corner in corners]
        
        # 如果检测到的角点太多，只保留最显著的4个
        if len(corners) > 4:
            # 按照到轮廓中心的距离排序
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                corners.sort(key=lambda p: ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5, reverse=True)
            corners = corners[:4]
        
        return corners
    
    def _calculate_corner_angles(self, corners):
        """计算角点的内角"""
        if len(corners) < 3:
            return []
        
        angles = []
        n = len(corners)
        for i in range(n):
            prev = (i - 1) % n
            curr = i
            next_idx = (i + 1) % n
            
            v1 = np.array(corners[prev]) - np.array(corners[curr])
            v2 = np.array(corners[next_idx]) - np.array(corners[curr])
            
            # 计算两向量夹角（度数）
            if np.linalg.norm(v1) * np.linalg.norm(v2) > 0:
                dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                dot = max(-1, min(1, dot))  # 确保在[-1, 1]范围内
                angle = np.degrees(np.arccos(dot))
                angles.append(angle)
        
        return angles
    
    def _is_rounded_rectangle(self, contour, approx, area):
        """检测是否为圆角矩形"""
        # 圆角矩形特征：与椭圆拟合度高，与矩形拟合度也较高
        if len(contour) < 5:
            return False
        
        try:
            # 拟合椭圆
            ellipse = cv2.fitEllipse(contour)
            ellipse_area = np.pi * (ellipse[1][0]/2) * (ellipse[1][1]/2)
            
            # 计算椭圆面积与轮廓面积的比例
            area_diff_ratio = abs(area - ellipse_area) / area if area > 0 else 1
            
            # 计算与矩形的相似度
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            rect_area_ratio = area / rect_area if rect_area > 0 else 0
            
            # 圆角矩形的特征：
            # 1. 与椭圆有一定差异（不是纯椭圆）
            # 2. 与矩形有一定相似性（基本是矩形形状）
            # 3. 点数较多（有圆角）
            if 0.2 < area_diff_ratio < 0.5 and rect_area_ratio > 0.7 and len(contour) > 10:
                return True
            
            # 另一种判断方法：判断角点附近是否有圆弧
            if len(approx) == 4 and rect_area_ratio > 0.8:
                # 提取四个角点
                corners = []
                for point in approx:
                    corners.append((point[0][0], point[0][1]))
                
                # 检查角点附近是否有多个轮廓点（圆弧特征）
                corner_regions_complex = 0
                for corner in corners:
                    points_near_corner = 0
                    for point in contour:
                        point = point[0]
                        dist = ((point[0] - corner[0])**2 + (point[1] - corner[1])**2)**0.5
                        if dist < min(w, h) * 0.2:  # 角点周围20%区域
                            points_near_corner += 1
                    
                    # 如果角点附近有多个点，可能是圆弧
                    if points_near_corner > 5:
                        corner_regions_complex += 1
                
                # 如果至少有3个角是圆弧，判断为圆角矩形
                if corner_regions_complex >= 3:
                    return True
            
            return False
        except:
            return False
    
    def _detect_shapes_with_hough(self, binary_image, original_image):
        """使用霍夫变换检测直线和矩形"""
        shapes = []
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(binary_image, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return shapes
        
        # 创建调试图像
        debug_image = original_image.copy()
        
        # 查找可能的矩形
        rectangles = []
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 计算线段中点
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            
            # 查找与当前线段垂直的线段
            for j, other_line in enumerate(lines):
                if i == j:
                    continue
                    
                x3, y3, x4, y4 = other_line[0]
                
                # 计算两条线段的方向向量
                dx1 = x2 - x1
                dy1 = y2 - y1
                dx2 = x4 - x3
                dy2 = y4 - y3
                
                # 计算点积，判断是否垂直
                dot_product = dx1 * dx2 + dy1 * dy2
                
                # 如果两条线段近似垂直
                if abs(dot_product) < 0.3 * (dx1**2 + dy1**2) * (dx2**2 + dy2**2)**0.5:
                    # 计算两条线段的交点
                    # 这里使用简化的方法，取两条线段的中点
                    mid_x2 = (x3 + x4) // 2
                    mid_y2 = (y3 + y4) // 2
                    
                    # 计算可能的矩形中心
                    center_x = (mid_x + mid_x2) // 2
                    center_y = (mid_y + mid_y2) // 2
                    
                    # 计算可能的矩形宽高
                    width = max(abs(x2 - x1), abs(x4 - x3))
                    height = max(abs(y2 - y1), abs(y4 - y3))
                    
                    # 创建矩形
                    rect = {
                        "center": (center_x, center_y),
                        "width": width,
                        "height": height
                    }
                    
                    rectangles.append(rect)
        
        # 合并重叠的矩形
        merged_rectangles = []
        for rect in rectangles:
            # 检查是否与已有矩形重叠
            is_merged = False
            for i, merged_rect in enumerate(merged_rectangles):
                # 计算两个矩形中心点之间的距离
                dist = ((rect["center"][0] - merged_rect["center"][0])**2 + 
                        (rect["center"][1] - merged_rect["center"][1])**2)**0.5
                
                # 如果距离小于矩形尺寸，认为是重叠的
                if dist < (rect["width"] + merged_rect["width"] + rect["height"] + merged_rect["height"]) / 4:
                    # 合并矩形
                    merged_rect["center"] = ((merged_rect["center"][0] + rect["center"][0]) // 2,
                                            (merged_rect["center"][1] + rect["center"][1]) // 2)
                    merged_rect["width"] = max(merged_rect["width"], rect["width"])
                    merged_rect["height"] = max(merged_rect["height"], rect["height"])
                    is_merged = True
                    break
            
            if not is_merged:
                merged_rectangles.append(rect)
        
        # 将合并后的矩形转换为Shape对象
        for i, rect in enumerate(merged_rectangles):
            center_x, center_y = rect["center"]
            width, height = rect["width"], rect["height"]
            
            # 创建矩形轮廓
            x1 = center_x - width // 2
            y1 = center_y - height // 2
            x2 = center_x + width // 2
            y2 = center_y + height // 2
            
            contour = np.array([
                [[x1, y1]],
                [[x2, y1]],
                [[x2, y2]],
                [[x1, y2]]
            ], dtype=np.int32)
            
            # 创建Shape对象
            shape = {
                "type": "rectangle",
                "contour": contour.tolist(),
                "bbox": (x1, y1, width, height),
                "center": (center_x, center_y),
                "area": width * height,
                "confidence": 0.7,
                "position": {"x": center_x, "y": center_y},
                "text": ""  # 初始化文本字段
            }
            
            # 在调试图像上绘制矩形
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(debug_image, f"Rect {i}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            shapes.append(shape)
        
        # 保存霍夫变换调试图像
        debug_path = os.path.join(app.config['UPLOAD_FOLDER'], f"hough_debug_{uuid.uuid4()}.jpg")
        cv2.imwrite(debug_path, debug_image)
        print(f"霍夫变换调试图像已保存: {debug_path}")
        
        return shapes

class RealOCREngine:
    """实际的OCR引擎"""
    def __init__(self, lang='ch'):
        # 使用静默模式初始化PaddleOCR，设置更低的置信度阈值以捕获更多文本
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False, 
                            det_db_thresh=0.3, # 降低检测阈值
                            det_db_box_thresh=0.3, # 降低框阈值
                            rec_model_dir=None, # 使用默认模型
                            det_model_dir=None, # 使用默认模型
                            cls_model_dir=None) # 使用默认模型
        self.min_confidence = 0.3  # 降低最小置信度阈值，以捕获更多文本
    
    def extract_text(self, image):
        """从图像中提取文本"""
        # 确保图像是BGR格式
        if len(image.shape) == 2:  # 灰度图
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 创建调试图像
        debug_image = image.copy()
        
        # 图像预处理：增强对比度
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 运行OCR
        result = self.ocr.ocr(enhanced_image, cls=True)
        
        # 处理结果
        text_regions = []
        
        if result is None or len(result) == 0:
            print("OCR未检测到任何文本")
            return text_regions
        
        # PaddleOCR返回格式: [[[x1,y1],[x2,y1],[x2,y2],[x1,y2]], (text, confidence)]
        for i, line in enumerate(result[0]):
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
            text_region = {
                "text": text,
                "bbox": (x_min, y_min, width, height),
                "confidence": confidence,
                "center": (x_min + width // 2, y_min + height // 2),
                "position": {"x": x_min + width // 2, "y": y_min + height // 2}
            }
            
            # 在调试图像上绘制文本区域
            cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(debug_image, f"{i}: {text} ({confidence:.2f})", 
                       (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            text_regions.append(text_region)
        
        # 保存调试图像
        debug_path = os.path.join(app.config['UPLOAD_FOLDER'], f"ocr_debug_{uuid.uuid4()}.jpg")
        cv2.imwrite(debug_path, debug_image)
        print(f"OCR调试图像已保存: {debug_path}")
        
        print(f"OCR识别到 {len(text_regions)} 个文本区域")
        
        return text_regions

class RealAIEvaluator:
    """实际的AI评估器"""
    def __init__(self):
        # 从环境变量获取API密钥
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError("未设置有效的DASHSCOPE_API_KEY环境变量")
        
        try:
            # 初始化OpenAI客户端（使用百炼API）
            import openai
            openai.api_key = self.api_key
            openai.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.openai = openai
            print("OpenAI客户端初始化成功")
        except Exception as e:
            print(f"OpenAI客户端初始化失败: {str(e)}")
            raise
    
    def evaluate(self, structured_data, image_path):
        """评估流程图"""
        # 准备评估提示
        prompt = self._prepare_evaluation_prompt(structured_data)
        
        # 读取图像并转换为base64
        image_base64 = self._encode_image(image_path)
        
        try:
            # 调用AI模型进行评估
            response = self.openai.ChatCompletion.create(
                model="qwen-vl-max",  # 使用qwen-vl模型
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ]
            )
            
            # 解析评估结果
            evaluation_result = self._parse_evaluation_result(response.choices[0].message.content)
            
            # 添加统计信息
            evaluation_result["statistics"] = structured_data.get("statistics", {})
            
            return evaluation_result
            
        except Exception as e:
            # 如果API调用失败，返回错误信息
            print(f"AI评估失败: {str(e)}")
            return {
                "error": str(e),
                "scores": {
                    "逻辑性": 0,
                    "合理性": 0,
                    "清晰度": 0,
                    "创新性": 0,
                    "总分": 0
                },
                "feedback": f"AI评估失败: {str(e)}，请稍后重试。",
                "statistics": structured_data.get("statistics", {})
            }
    
    def _prepare_evaluation_prompt(self, structured_data):
        """准备评估提示"""
        # 将结构化数据转换为JSON字符串
        data_json = json.dumps(structured_data, ensure_ascii=False, indent=2)
        
        # 构建提示
        prompt = """你是一个教学流程图评估专家。请根据以下规则评估这张教学流程图：

## 标准符号规则
- 矩形框：表示教学活动或步骤（如"教师讲解"）
- 平行四边形：表示输入/输出（如"学生提交作业"）
- 菱形框：表示决策或分支（如"是否达标？"）
- 箭头：表示流程方向
- 圆角矩形：表示起点或终点（如"开始""结束"）
- 基本规范：一个矩形、平行四边形、圆角矩形（中间有竖线）三者成模块出现

## 基本规范
- 条件判断：分支流程要有明确的条件判断（如"是否完成任务→是/否"）
- 时间标注（可选）：在步骤旁标注预计时间分配（如"小组讨论：5分钟"）
- 流程方向：自上而下，不能交叉连线

## 评价标准
- 逻辑性：步骤顺序要符合教学逻辑（如"导入→讲授→练习→总结"）
- 合理性：能区分教师活动（如"讲解案例"）与学生活动（如"分组实验"）
- 清晰度：能用简练语言描述活动（如"案例导入"而非"教师使用案例导入新课"）
- 创新性：是否融入非传统技术或教学模式（如翻转课堂、项目式学习）

以下是流程图的结构化数据：
```
""" + data_json + """
```

请分析图像和结构化数据，并给出以下评估结果：
1. 对每个评价标准（逻辑性、合理性、清晰度、创新性）的得分（1-10分）
2. 对流程图的整体评价（1-100分）
3. 详细的反馈意见，包括优点和改进建议
4. 是否符合标准符号规则和基本规范

请以JSON格式返回结果，格式如下：
```json
{
  "scores": {
    "逻辑性": 
    "合理性": 
    "清晰度": 
    "创新性": 
    "总分": 
  },
  "feedback": "详细的反馈意见",
  "符合规范": true,
  "改进建议": ["建议1", "建议2", "建议3"]
}
```"""
        
        return prompt
    
    def _encode_image(self, image_path):
        """将图像编码为base64字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _parse_evaluation_result(self, response_text):
        """解析模型返回的评估结果"""
        try:
            # 尝试从响应文本中提取JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                return result
            
            # 如果没有找到JSON，返回原始文本作为反馈
            return {
                "scores": {
                    "逻辑性": 0,
                    "合理性": 0,
                    "清晰度": 0,
                    "创新性": 0,
                    "总分": 0
                },
                "feedback": response_text,
                "符合规范": False,
                "改进建议": []
            }
            
        except Exception as e:
            # 解析失败，返回错误信息
            return {
                "scores": {
                    "逻辑性": 0,
                    "合理性": 0,
                    "清晰度": 0,
                    "创新性": 0,
                    "总分": 0
                },
                "feedback": f"无法解析评估结果: {str(e)}",
                "符合规范": False,
                "改进建议": []
            }

class QwenVLOCREngine:
    """使用千问视觉OCR API的OCR引擎"""
    def __init__(self):
        # 从环境变量获取API密钥
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key or self.api_key == "your_api_key_here":
            print("警告: 未设置有效的DASHSCOPE_API_KEY环境变量，千问视觉OCR将不可用")
            self.client = None
            return
        
        try:
            # 初始化OpenAI客户端（使用百炼API）
            import openai
            openai.api_key = self.api_key
            openai.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.client = openai
            print("千问视觉OCR客户端初始化成功")
            print(f"使用API密钥: {self.api_key[:5]}...{self.api_key[-5:]}")
        except Exception as e:
            print(f"千问视觉OCR客户端初始化失败: {str(e)}")
            self.client = None
    
    def extract_text(self, image):
        """从图像中提取文本"""
        # 如果客户端未初始化，返回空结果
        if self.client is None:
            print("千问视觉OCR客户端未初始化，无法提取文本")
            return []
            
        # 创建调试图像
        debug_image = image.copy()
        
        # 保存临时图像文件
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}.jpg")
        cv2.imwrite(temp_image_path, image)
        
        # 将图像转换为base64
        try:
            with open(temp_image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"图像转换为base64时出错: {str(e)}")
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            return []
        
        try:
            print("正在调用千问视觉OCR API...")
            # 调用千问视觉OCR API - 使用更明确的提示，要求识别每个图形元素中的文字
            completion = self.client.ChatCompletion.create(
                model="qwen-vl-ocr-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "min_pixels": 28 * 28 * 4,
                                    "max_pixels": 1280 * 784
                                }
                            },
                            {"type": "text", "text": "Identify all text in this flowchart. For each shape or element in the flowchart, extract the text inside it. List each text element on a separate line with its position."}
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # 解析OCR结果
            if not hasattr(completion, 'choices') or not completion.choices or not hasattr(completion.choices[0], 'message') or not hasattr(completion.choices[0].message, 'content'):
                print("千问视觉OCR返回结果格式异常")
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                return []
                
            response_content = completion.choices[0].message.content
            print(f"千问视觉OCR响应: {response_content}")
            
            # 解析OCR结果中的文本区域
            text_regions = self._parse_ocr_result(response_content, image.shape[1], image.shape[0])
            
            # 在调试图像上绘制文本区域
            # for i, text_region in enumerate(text_regions):
            #     x_min, y_min, width, height = text_region["bbox"]
            #     x_max = x_min + width
            #     y_max = y_min + height
                
            #     cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            #     cv2.putText(debug_image, f"{i}: {text_region['text']}", 
            #                (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 保存调试图像
            debug_path = os.path.join(app.config['UPLOAD_FOLDER'], f"qwen_ocr_debug_{uuid.uuid4()}.jpg")
            cv2.imwrite(debug_path, debug_image)
            print(f"千问视觉OCR调试图像已保存: {debug_path}")
            
            # 删除临时文件
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            
            print(f"千问视觉OCR识别到 {len(text_regions)} 个文本区域")
            
            return text_regions
            
        except Exception as e:
            print(f"千问视觉OCR调用失败: {str(e)}")
            # 删除临时文件
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            return []
    
    def _parse_ocr_result(self, response_content, image_width, image_height):
        """解析OCR结果，提取文本区域"""
        text_regions = []
        
        try:
            # 按行分割响应内容
            lines = response_content.split('\n')
            
            # 将所有文本合并为一个字符串
            all_text = ' '.join([line.strip() for line in lines if line.strip()])
            
            # 直接按空格分割文本
            words = all_text.split()
            
            # 需要过滤的关键词
            filtered_words = ["是", "否"]
            
            # 为每个分割后的词创建一个文本区域
            for i, word in enumerate(words):
                if not word.strip():
                    continue
                
                # 过滤掉"是"或"否"的文本
                if word.strip() in filtered_words:
                    print(f"过滤掉关键词: {word}")
                    continue
                    
                # 估算每个文本在图像中的位置
                position = self._estimate_element_position(word, i, len(words), image_width, image_height)
                
                text_region = {
                    "id": f"text_{i}",
                    "text": word,
                    "bbox": position,
                    "position": {
                        "x": (position[0] + position[2]) // 2,
                        "y": (position[1] + position[3]) // 2
                    },
                    "confidence": 0.9  # 默认置信度
                }
                
                text_regions.append(text_region)
            
            return text_regions
            
        except Exception as e:
            print(f"解析OCR结果失败: {str(e)}")
            return []
    
    def _estimate_element_position(self, element, index, total_elements, image_width, image_height):
        """根据元素内容估计其在流程图中的位置"""
        # 默认文本大小
        text_width = len(element) * 10
        text_height = 30
        
        # 根据元素内容确定可能的位置
        element_lower = element.lower()
        
        # 开始节点通常在顶部
        if "start" in element_lower or "begin" in element_lower or "开始" in element_lower:
            x = (image_width - text_width) // 2
            y = int(image_height * 0.1)
        
        # 结束节点通常在底部
        elif "end" in element_lower or "finish" in element_lower or "结束" in element_lower:
            x = (image_width - text_width) // 2
            y = int(image_height * 0.9)
        
        # 输入节点通常在顶部附近
        elif "input" in element_lower or "输入" in element_lower:
            x = (image_width - text_width) // 2
            y = int(image_height * 0.2)
        
        # 输出节点通常在底部附近
        elif "output" in element_lower or "输出" in element_lower:
            x = (image_width - text_width) // 2
            y = int(image_height * 0.8)
        
        # 处理节点通常在中间
        elif "process" in element_lower or "处理" in element_lower or "计算" in element_lower:
            x = (image_width - text_width) // 2
            y = int(image_height * 0.4)
        
        # 决策节点通常在中间
        elif "decision" in element_lower or "判断" in element_lower or "是否" in element_lower or "if" in element_lower or "?" in element:
            x = (image_width - text_width) // 2
            y = int(image_height * 0.5)
        
        # 是/否通常在决策节点附近
        elif element_lower in ["yes", "no", "是", "否", "true", "false"]:
            if element_lower in ["yes", "是", "true"]:
                x = int(image_width * 0.6)
                y = int(image_height * 0.55)
            else:  # no, 否, false
                x = int(image_width * 0.3)
                y = int(image_height * 0.45)
        
        # 错误处理通常在左侧或右侧
        elif "error" in element_lower or "错误" in element_lower or "异常" in element_lower:
            x = int(image_width * 0.2)
            y = int(image_height * 0.6)
        
        # 其他元素均匀分布
        else:
            # 计算在流程图中的相对位置
            relative_pos = (index + 1) / (total_elements + 1)
            
            # 如果元素较少，垂直排列
            if total_elements <= 5:
                x = (image_width - text_width) // 2
                y = int(image_height * relative_pos)
            # 如果元素较多，尝试网格排列
            else:
                cols = min(3, total_elements // 2 + 1)
                col = index % cols
                row = index // cols
                
                x = int(image_width * (col + 1) / (cols + 1) - text_width / 2)
                y = int(image_height * (row + 1) / (total_elements // cols + 2))
        
        # 确保边界框在图像内
        x = max(10, min(x, image_width - text_width - 10))
        y = max(10, min(y, image_height - text_height - 10))
        
        return x, y, text_width, text_height

def draw_chinese_text(img, text, position, font_size=20, color=(0, 0, 0), thickness=2):
    """使用PIL绘制中文文本到OpenCV图像上"""
    # 将OpenCV图像转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 创建绘图对象
    draw = ImageDraw.Draw(img_pil)
    
    # 设置字体
    try:
        # 尝试使用系统字体
        font_path = "C:/Windows/Fonts/simhei.ttf"  # Windows系统黑体
        if not os.path.exists(font_path):
            # Linux系统字体
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # 如果系统字体不可用，使用默认字体
        font = ImageFont.load_default()
    
    # 绘制文本
    draw.text(position, text, font=font, fill=color)
    
    # 将PIL图像转换回OpenCV图像
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return img_cv

def visualize_results(image, shapes, text_regions, nodes):
    """将识别结果可视化"""
    # 创建图像副本
    vis_image = image.copy()
    
    # 确保输入数据有效
    if not shapes:
        print("警告: 没有图形可视化")
    if not text_regions:
        print("警告: 没有文本区域可视化")
    if not nodes:
        print("警告: 没有节点可视化")
    
    # 绘制图形
    for i, shape in enumerate(shapes):
        try:
            # 获取边界框
            if "bbox" not in shape:
                print(f"警告: 图形 {i} 没有边界框，跳过")
                continue
                
            bbox = shape["bbox"]
            # 确保bbox是元组或列表，并且有4个元素
            if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                print(f"警告: 图形 {i} 的边界框格式无效: {bbox}，跳过")
                continue
                
            x, y, w, h = bbox
            
            # 根据图形类型设置颜色
            if shape["type"] == "矩形":
                color = (0, 255, 0)  # 绿色
            elif shape["type"] == "菱形":
                color = (0, 0, 255)  # 红色
            elif shape["type"] == "圆角矩形":
                color = (255, 0, 0)  # 蓝色
            elif shape["type"] == "平行四边形":
                color = (0, 255, 255)  # 黄色
            else:
                color = (128, 128, 128)  # 灰色
            
            # 绘制轮廓
            if "contour" in shape and shape["contour"]:
                try:
                    contour = np.array(shape["contour"], dtype=np.int32)
                    cv2.drawContours(vis_image, [contour], 0, color, 2)
                except Exception as e:
                    print(f"绘制轮廓时出错: {e}")
            
            # 确保shape有center属性
            if "center" not in shape:
                if "position" in shape and "x" in shape["position"] and "y" in shape["position"]:
                    shape["center"] = (shape["position"]["x"], shape["position"]["y"])
                else:
                    print(f"警告: 图形 {i} 缺少center和position属性，跳过中心点绘制")
                    continue
                    
            # 绘制中心点
            center = shape["center"]
            cv2.circle(vis_image, center, 5, color, -1)
            
            # 绘制图形ID和类型标签 - 使用中文绘制函数
            label = f"图形 {i}: {shape['type']}"
            # 不再使用cv2.putText，改用我们的draw_chinese_text函数
            vis_image = draw_chinese_text(vis_image, label, (x, y-20), font_size=15, color=color, thickness=2)
            
            # 如果有关联文本，绘制连接线
            if "text" in shape and shape["text"]:
                # 查找对应的文本区域
                for text_region in text_regions:
                    if "text" in text_region and text_region["text"] == shape["text"]:
                        # 确保text_region有center属性
                        if "center" not in text_region:
                            if "position" in text_region and "x" in text_region["position"] and "y" in text_region["position"]:
                                text_region["center"] = (text_region["position"]["x"], text_region["position"]["y"])
                            elif "bbox" in text_region and len(text_region["bbox"]) == 4:
                                x_tr, y_tr, w_tr, h_tr = text_region["bbox"]
                                text_region["center"] = (x_tr + w_tr // 2, y_tr + h_tr // 2)
                            else:
                                print(f"警告: 文本区域缺少center、position和bbox属性，跳过连接线绘制")
                                continue
                                
                        text_center = text_region["center"]
                        # 绘制连接线
                        # cv2.line(vis_image, center, text_center, color, 1, cv2.LINE_AA)
                        break
        except Exception as e:
            print(f"处理图形 {i} 时出错: {e}")
    
    # 绘制文本区域
    # for i, text_region in enumerate(text_regions):
        # try:
        #     # 获取边界框
        #     if "bbox" not in text_region:
        #         print(f"警告: 文本区域 {i} 没有边界框，跳过")
        #         continue
                
        #     bbox = text_region["bbox"]
        #     # 确保bbox是元组或列表，并且有4个元素
        #     if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        #         print(f"警告: 文本区域 {i} 的边界框格式无效: {bbox}，跳过")
        #         continue
                
        #     x, y, w, h = bbox
            
            # 绘制边界框
            # cv2.rectangle(vis_image, (x, y), (x+w, y+h), (255, 0, 255), 2)  # 紫色
            
            # # 绘制文本ID和内容
            # if "text" in text_region:
            #     label = f"Text {i}: {text_region['text']}"
        #     #     cv2.putText(vis_image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
        #     # 如果有置信度，显示置信度
        #     if "confidence" in text_region:
        #         conf_label = f"{text_region['confidence']:.2f}"
        #         cv2.putText(vis_image, conf_label, (x+w-30, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
        #     # 确保text_region有center属性
        #     if "center" not in text_region:
        #         if "position" in text_region and "x" in text_region["position"] and "y" in text_region["position"]:
        #             text_region["center"] = (text_region["position"]["x"], text_region["position"]["y"])
        #         else:
        #             text_region["center"] = (x + w // 2, y + h // 2)
        #             if "position" not in text_region:
        #                 text_region["position"] = {"x": x + w // 2, "y": y + h // 2}
                
        #     # 绘制中心点
        #     cv2.circle(vis_image, text_region["center"], 3, (255, 0, 255), -1)
        # except Exception as e:
        #     print(f"处理文本区域 {i} 时出错: {e}")
    
    # # 绘制节点
    # for node in nodes:
    #     try:
    #         # 获取位置
    #         if "position" not in node or "x" not in node["position"] or "y" not in node["position"]:
    #             print(f"警告: 节点 {node.get('id', '?')} 的位置无效，跳过")
    #             continue
                
    #         x, y = node["position"]["x"], node["position"]["y"]
            
    #         # 绘制节点ID和类型
    #         label = f"Node {node.get('id', '?')}: {node.get('type', '?')}"
    #         cv2.putText(vis_image, label, (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
    #         # 绘制连接关系
    #         if "connections" in node and node["connections"]:
    #             for conn_id in node["connections"]:
    #                 if conn_id < len(nodes):
    #                     conn_node = nodes[conn_id]
    #                     if "position" in conn_node and "x" in conn_node["position"] and "y" in conn_node["position"]:
    #                         conn_x, conn_y = conn_node["position"]["x"], conn_node["position"]["y"]
    #                         # 绘制箭头
    #                         cv2.arrowedLine(vis_image, (x, y), (conn_x, conn_y), (0, 0, 0), 1, cv2.LINE_AA, tipLength=0.03)
    #     except Exception as e:
    #         print(f"处理节点 {node.get('id', '?')} 时出错: {e}")
    
    # 添加图例
    try:
        legend_y = 30
        # 矩形
        cv2.rectangle(vis_image, (10, legend_y), (30, legend_y+20), (0, 255, 0), -1)
        vis_image = draw_chinese_text(vis_image, "矩形", (35, legend_y), font_size=15, color=(0, 0, 0), thickness=1)
        
        # 菱形
        legend_y += 30
        pts = np.array([[20, legend_y], [30, legend_y+10], [20, legend_y+20], [10, legend_y+10]], np.int32)
        cv2.fillPoly(vis_image, [pts], (0, 0, 255))
        vis_image = draw_chinese_text(vis_image, "菱形", (35, legend_y), font_size=15, color=(0, 0, 0), thickness=1)
        
        # 圆角矩形
        legend_y += 30
        cv2.ellipse(vis_image, (20, legend_y+10), (10, 10), 0, 0, 360, (255, 0, 0), -1)
        vis_image = draw_chinese_text(vis_image, "圆角矩形", (35, legend_y), font_size=15, color=(0, 0, 0), thickness=1)
        
        # 平行四边形
        legend_y += 30
        pts = np.array([[10, legend_y+20], [25, legend_y+20], [30, legend_y], [15, legend_y]], np.int32)
        cv2.fillPoly(vis_image, [pts], (0, 255, 255))
        vis_image = draw_chinese_text(vis_image, "平行四边形", (35, legend_y), font_size=15, color=(0, 0, 0), thickness=1)
        
        # 文本
        legend_y += 30
        cv2.rectangle(vis_image, (10, legend_y), (30, legend_y+20), (255, 0, 255), -1)
        vis_image = draw_chinese_text(vis_image, "文本", (35, legend_y), font_size=15, color=(0, 0, 0), thickness=1)
        
        # 添加统计信息
        stats_y = legend_y + 50
        vis_image = draw_chinese_text(vis_image, f"图形数量: {len(shapes)}", (10, stats_y), font_size=18, color=(0, 0, 0), thickness=2)
        stats_y += 25
        vis_image = draw_chinese_text(vis_image, f"文本数量: {len(text_regions)}", (10, stats_y), font_size=18, color=(0, 0, 0), thickness=2)
        stats_y += 25
        vis_image = draw_chinese_text(vis_image, f"节点数量: {len(nodes)}", (10, stats_y), font_size=18, color=(0, 0, 0), thickness=2)
    except Exception as e:
        print(f"绘制图例时出错: {e}")
    
    # 保存可视化结果
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"vis_{uuid.uuid4()}.jpg")
    cv2.imwrite(output_path, vis_image)
    
    # 返回相对路径
    return os.path.basename(output_path)

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    if 'file' not in request.files:
        flash('没有选择文件')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('没有选择文件')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # 生成唯一文件名
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 处理流程图
        try:
            # 1. 图像预处理和分割
            if CV_AVAILABLE:
                image_processor = RealImageProcessor()
            else:
                image_processor = MockImageProcessor()
            processed_image, shapes = image_processor.process(filepath)
            
            # 2. OCR文本识别 - 强制使用千问视觉OCR
            if CV_AVAILABLE:
                # 使用千问视觉OCR
                ocr_engine = QwenVLOCREngine()
                text_regions = ocr_engine.extract_text(processed_image)
                if not text_regions:  # 如果千问OCR失败，使用模拟OCR
                    print("千问视觉OCR未返回结果，使用模拟OCR")
                    ocr_engine = MockOCREngine()
                    text_regions = ocr_engine.extract_text(processed_image)
            else:
                # 使用模拟OCR
                ocr_engine = MockOCREngine()
                text_regions = ocr_engine.extract_text(None)
            
            # 3. 图形-文本关联
            if CV_AVAILABLE:
                flowchart_analyzer = RealFlowchartAnalyzer()
            else:
                flowchart_analyzer = MockFlowchartAnalyzer()
            structured_data = flowchart_analyzer.analyze(shapes, text_regions)
            
            # 4. 可视化识别结果
            if CV_AVAILABLE:
                vis_filename = visualize_results(processed_image, shapes, text_regions, structured_data["nodes"])
            else:
                vis_filename = None
            
            # 5. AI评估
            if OPENAI_AVAILABLE:
                try:
                    ai_evaluator = RealAIEvaluator()
                    evaluation_result = ai_evaluator.evaluate(structured_data, filepath)
                except ValueError as e:
                    print(f"AI评估器初始化失败: {str(e)}")
                    ai_evaluator = MockAIEvaluator()
                    evaluation_result = ai_evaluator.evaluate(structured_data, filepath)
            else:
                ai_evaluator = MockAIEvaluator()
                evaluation_result = ai_evaluator.evaluate(structured_data, filepath)
            
            # 6. 规则验证
            try:
                print("\n============ 开始规则验证 ============")
                print(f"验证节点数量: {len(structured_data['nodes'])}")
                
                # 打印节点信息
                for i, node in enumerate(structured_data["nodes"]):
                    print(f"节点 {i+1}:")
                    print(f"  类型: {node.get('shape_type', 'unknown')}")
                    print(f"  文本: {node.get('text', 'empty')}")
                
                # 优先尝试使用千问API验证
                api_failed = False
                try:
                    validator = QwenValidator(Config.QWEN_API_KEY)
                    print(f"使用API密钥: {Config.QWEN_API_KEY[:8]}...")
                    
                    validation_result = validator.validate_shape_text(structured_data["nodes"])
                    print(f"千问API验证结果: {validation_result}")
                    
                    if validation_result['success']:
                        validation = validation_result['result']
                        print("千问API验证成功，获取到结果")
                    else:
                        print(f"千问API验证失败: {validation_result['error']}，标记为使用模拟验证器")
                        api_failed = True
                except Exception as e:
                    print(f"千问API验证异常: {str(e)}，标记为使用模拟验证器")
                    api_failed = True
                
                # 如果千问API验证失败，才使用模拟验证器
                if api_failed:
                    print("使用模拟验证器作为备用...")
                    mock_validator = MockValidator()
                    mock_result = mock_validator.validate_shape_text(structured_data["nodes"])
                    if mock_result["success"]:
                        validation = mock_result["result"]
                        print("模拟验证成功")
                    else:
                        validation = None
                        print(f"模拟验证失败: {mock_result['error']}")
            except Exception as e:
                validation = None
                import traceback
                print(f"规则验证异常: {str(e)}")
                print(traceback.format_exc())
                
            print("============ 规则验证结束 ============\n")
            
            # 返回处理结果
            return render_template('result.html', 
                                  image_path=f'uploads/{filename}',
                                  vis_image_path=f'uploads/{vis_filename}' if vis_filename else None,
                                  result=evaluation_result,
                                  shapes=shapes,
                                  text_regions=text_regions,
                                  nodes=structured_data["nodes"],
                                  validation=validation)
        
        except Exception as e:
            flash(f'处理失败: {str(e)}')
            return redirect(url_for('index'))
    
    flash('不支持的文件类型')
    return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API端点，用于处理流程图分析请求"""
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # 生成唯一文件名
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 处理流程图
            if CV_AVAILABLE:
                image_processor = RealImageProcessor()
            else:
                image_processor = MockImageProcessor()
            processed_image, shapes = image_processor.process(filepath)
            
            # 强制使用千问视觉OCR
            if CV_AVAILABLE:
                # 使用千问视觉OCR
                ocr_engine = QwenVLOCREngine()
                text_regions = ocr_engine.extract_text(processed_image)
                if not text_regions:  # 如果千问OCR失败，使用模拟OCR
                    print("千问视觉OCR未返回结果，使用模拟OCR")
                    ocr_engine = MockOCREngine()
                    text_regions = ocr_engine.extract_text(processed_image)
            else:
                # 使用模拟OCR
                ocr_engine = MockOCREngine()
                text_regions = ocr_engine.extract_text(None)
            
            if CV_AVAILABLE:
                flowchart_analyzer = RealFlowchartAnalyzer()
            else:
                flowchart_analyzer = MockFlowchartAnalyzer()
            structured_data = flowchart_analyzer.analyze(shapes, text_regions)
            
            if OPENAI_AVAILABLE:
                try:
                    ai_evaluator = RealAIEvaluator()
                    evaluation_result = ai_evaluator.evaluate(structured_data, filepath)
                except ValueError as e:
                    ai_evaluator = MockAIEvaluator()
                    evaluation_result = ai_evaluator.evaluate(structured_data, filepath)
            else:
                ai_evaluator = MockAIEvaluator()
                evaluation_result = ai_evaluator.evaluate(structured_data, filepath)
            
            # 规则验证
            try:
                print("\n============ API规则验证 ============")
                print(f"API验证节点数量: {len(structured_data['nodes'])}")
                
                # 优先尝试使用千问API验证
                api_failed = False
                try:
                    validator = QwenValidator(Config.QWEN_API_KEY)
                    validation_result = validator.validate_shape_text(structured_data["nodes"])
                    print(f"API千问验证结果: {validation_result}")
                    
                    if validation_result['success']:
                        validation = validation_result['result']
                        print("API千问验证成功，获取到结果")
                    else:
                        print(f"API千问验证失败: {validation_result['error']}，标记为使用模拟验证器")
                        api_failed = True
                except Exception as e:
                    print(f"API千问验证异常: {str(e)}，标记为使用模拟验证器")
                    api_failed = True
                
                # 如果千问API验证失败，使用模拟验证器
                if api_failed:
                    print("API使用模拟验证器作为备用...")
                    mock_validator = MockValidator()
                    mock_result = mock_validator.validate_shape_text(structured_data["nodes"])
                    if mock_result["success"]:
                        validation = mock_result["result"]
                        print("API模拟验证成功")
                    else:
                        validation = None
                        print(f"API模拟验证失败: {mock_result['error']}")
            except Exception as e:
                validation = None
                import traceback
                print(f"API规则验证异常: {str(e)}")
                print(traceback.format_exc())
                
            print("============ API规则验证结束 ============\n")
            
            return jsonify({
                'image_url': url_for('static', filename=f'uploads/{filename}', _external=True),
                'result': evaluation_result,
                'validation': validation
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': '不支持的文件类型'}), 400

from qwen_validator import QwenValidator

class MockValidator:
    """模拟验证器，在API不可用时提供基本验证"""
    
    def validate_shape_text(self, nodes):
        print("使用模拟验证器...")
        if not nodes:
            return {
                "success": False,
                "error": "节点数据为空"
            }
            
        details = []
        score = 0
        valid_count = 0
        
        for node in nodes:
            shape_type = node.get('shape_type', 'unknown')
            text = node.get('text', '')
            is_valid = False
            reason = ""
            suggestion = ""
            
            # 基本验证逻辑
            if shape_type == 'rectangle':
                # 不再使用预设关键词判断
                # 默认认为大部分图形符合规则，只在明显不符合时提供建议
                is_valid = True
                reason = "矩形一般表示教学活动或步骤，具体判断需由专业人员确认"
                suggestion = ""
            elif shape_type == 'parallelogram':
                is_valid = True
                reason = "平行四边形一般表示输入/输出，具体判断需由专业人员确认"
                suggestion = ""
            elif shape_type == 'diamond':
                is_valid = True
                reason = "菱形一般表示决策或分支，具体判断需由专业人员确认"
                suggestion = ""
            else:
                is_valid = False
                reason = f"未知的图形类型: {shape_type}，需要专业人员判断"
                suggestion = "建议使用标准图形符号（矩形、菱形、平行四边形等）"
            
            details.append({
                "shape_type": shape_type,
                "text": text,
                "is_valid": is_valid,
                "reason": reason,
                "suggestion": suggestion if not is_valid else ""
            })
            
            if is_valid:
                valid_count += 1
                
        # 计算总分
        if nodes:
            score = int((valid_count / len(nodes)) * 100)
            
        # 总体评价
        if score >= 80:
            feedback = "流程图符号使用规范，大部分图形与文本匹配良好"
        elif score >= 60:
            feedback = "流程图符号使用基本规范，但有一些图形与文本不匹配"
        else:
            feedback = "流程图符号使用不规范，大部分图形与文本不匹配"
            
        result = {
            "overall_valid": score >= 60,
            "score": score,
            "feedback": feedback,
            "details": details
        }
        
        return {
            "success": True,
            "result": result
        }

class Config:
    QWEN_API_KEY = "sk-3444250feb104de5acc16d911a91bb6f"  # 请替换为您的API密钥

@app.route('/api/validate_flowchart', methods=['POST'])
def validate_flowchart():
    try:
        data = request.get_json()
        if not data or 'nodes' not in data:
            return jsonify({'error': '未提供图形数据'}), 400
            
        nodes = data['nodes']
        if not nodes:
            return jsonify({'error': '图形数据为空'}), 400
            
        # 使用qwen验证图形和文本
        validator = QwenValidator(Config.QWEN_API_KEY)
        validation_result = validator.validate_shape_text(nodes)
        
        if not validation_result['success']:
            return jsonify({'error': validation_result['error']}), 500
            
        return jsonify({
            'validation': validation_result['result']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("教学流程图智能批阅系统启动中...")
    print(f"OpenCV 可用: {CV_AVAILABLE}")
    print(f"千问视觉OCR 可用: {QWEN_OCR_AVAILABLE}")
    print(f"OpenAI API 可用: {OPENAI_AVAILABLE}")
    
    if not CV_AVAILABLE:
        print("警告: OpenCV不可用，将使用模拟图像处理功能")
    
    if not QWEN_OCR_AVAILABLE:
        print("警告: 千问视觉OCR不可用，将使用模拟OCR功能")
        print("请在.env文件中设置有效的DASHSCOPE_API_KEY以启用千问视觉OCR")
    
    if not OPENAI_AVAILABLE:
        print("警告: OpenAI API不可用，将使用模拟AI评估功能")
    
    print("系统已配置为优先使用千问视觉OCR进行文本识别")
    app.run(debug=True) 