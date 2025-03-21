import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from app.utils.image_processor import Shape
from app.utils.ocr_engine import TextRegion

@dataclass
class FlowchartNode:
    """表示流程图中的一个节点"""
    id: int
    type: str  # 'start', 'end', 'activity', 'decision', 'input_output'
    text: str
    shape: Shape
    text_region: Optional[TextRegion] = None
    connections: List[int] = None  # 连接到的节点ID列表
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = []

@dataclass
class FlowchartConnection:
    """表示流程图中的一个连接"""
    from_node: int  # 起始节点ID
    to_node: int    # 目标节点ID
    label: str = ""  # 连接标签（如"是"/"否"）

class FlowchartAnalyzer:
    """分析流程图，关联图形和文本，构建结构化表示"""
    
    def __init__(self):
        self.distance_threshold = 50  # 图形和文本关联的距离阈值
        
        # 图形类型到流程图节点类型的映射
        self.shape_to_node_type = {
            'rectangle': 'activity',
            'diamond': 'decision',
            'ellipse': 'start_end',
            'parallelogram': 'input_output'
        }
    
    def analyze(self, shapes: List[Shape], text_regions: List[TextRegion]) -> Dict[str, Any]:
        """
        分析流程图，构建结构化表示
        
        Args:
            shapes: 检测到的图形列表
            text_regions: 识别的文本区域列表
            
        Returns:
            流程图的结构化表示
        """
        # 1. 关联图形和文本
        self._associate_text_with_shapes(shapes, text_regions)
        
        # 2. 创建流程图节点
        nodes = self._create_flowchart_nodes(shapes)
        
        # 3. 检测节点之间的连接
        connections = self._detect_connections(nodes)
        
        # 4. 构建结构化数据
        structured_data = {
            'nodes': [self._node_to_dict(node) for node in nodes],
            'connections': [self._connection_to_dict(conn) for conn in connections],
            'statistics': self._calculate_statistics(nodes)
        }
        
        return structured_data
    
    def _associate_text_with_shapes(self, shapes: List[Shape], text_regions: List[TextRegion]) -> None:
        """关联图形和文本"""
        for text_region in text_regions:
            closest_shape = None
            min_distance = float('inf')
            
            for shape in shapes:
                # 计算文本中心点到图形中心点的距离
                text_center = text_region.center
                shape_center = shape.center
                
                distance = np.sqrt((text_center[0] - shape_center[0])**2 + 
                                  (text_center[1] - shape_center[1])**2)
                
                # 检查文本是否在图形内部或附近
                if distance < min_distance:
                    min_distance = distance
                    closest_shape = shape
            
            # 如果找到最近的图形且距离在阈值内
            if closest_shape is not None and min_distance < self.distance_threshold:
                # 关联文本和图形
                text_region.associated_shape = closest_shape
                closest_shape.text_region = text_region.bbox
    
    def _create_flowchart_nodes(self, shapes: List[Shape]) -> List[FlowchartNode]:
        """创建流程图节点"""
        nodes = []
        
        for i, shape in enumerate(shapes):
            # 确定节点类型
            node_type = self.shape_to_node_type.get(shape.type, 'activity')
            
            # 特殊处理椭圆形（开始/结束）
            if node_type == 'start_end':
                # 如果有文本，根据文本内容确定是开始还是结束
                if hasattr(shape, 'text_region') and shape.text_region is not None:
                    text = self._get_text_from_region(shape.text_region)
                    if '开始' in text or '起点' in text or '开始' in text:
                        node_type = 'start'
                    elif '结束' in text or '终点' in text or '结束' in text:
                        node_type = 'end'
                    else:
                        node_type = 'start'  # 默认为开始
                else:
                    node_type = 'start'  # 默认为开始
            
            # 获取节点文本
            text = ""
            if hasattr(shape, 'text_region') and shape.text_region is not None:
                text = self._get_text_from_region(shape.text_region)
            
            # 创建节点
            node = FlowchartNode(
                id=i,
                type=node_type,
                text=text,
                shape=shape,
                text_region=None if not hasattr(shape, 'text_region') else shape.text_region
            )
            
            nodes.append(node)
        
        return nodes
    
    def _detect_connections(self, nodes: List[FlowchartNode]) -> List[FlowchartConnection]:
        """检测节点之间的连接"""
        connections = []
        
        # 简单的启发式方法：根据节点的相对位置确定连接
        # 在实际应用中，应该使用更复杂的算法检测箭头和连线
        
        # 按y坐标排序节点（从上到下）
        sorted_nodes = sorted(nodes, key=lambda node: node.shape.center[1])
        
        for i, node in enumerate(sorted_nodes):
            # 跳过最后一个节点
            if i == len(sorted_nodes) - 1:
                continue
            
            # 获取当前节点下方的节点
            next_nodes = [n for n in sorted_nodes[i+1:] 
                         if abs(n.shape.center[0] - node.shape.center[0]) < 100]
            
            if next_nodes:
                # 连接到最近的下方节点
                next_node = min(next_nodes, 
                               key=lambda n: abs(n.shape.center[1] - node.shape.center[1]))
                
                # 创建连接
                connection = FlowchartConnection(
                    from_node=node.id,
                    to_node=next_node.id
                )
                
                # 如果是决策节点，添加标签
                if node.type == 'decision':
                    # 左侧连接标记为"否"
                    if next_node.shape.center[0] < node.shape.center[0]:
                        connection.label = "否"
                    # 右侧连接标记为"是"
                    else:
                        connection.label = "是"
                
                connections.append(connection)
                
                # 更新节点的连接列表
                node.connections.append(next_node.id)
        
        return connections
    
    def _calculate_statistics(self, nodes: List[FlowchartNode]) -> Dict[str, int]:
        """计算流程图统计信息"""
        # 统计各类型节点数量
        node_types = {}
        for node in nodes:
            if node.type in node_types:
                node_types[node.type] += 1
            else:
                node_types[node.type] = 1
        
        # 计算总节点数
        total_nodes = len(nodes)
        
        return {
            'total_nodes': total_nodes,
            'node_types': node_types
        }
    
    def _get_text_from_region(self, region) -> str:
        """从文本区域获取文本内容"""
        if isinstance(region, TextRegion):
            return region.text
        return ""
    
    def _node_to_dict(self, node: FlowchartNode) -> Dict[str, Any]:
        """将节点转换为字典表示"""
        return {
            'id': node.id,
            'type': node.type,
            'text': node.text,
            'position': {
                'x': node.shape.center[0],
                'y': node.shape.center[1]
            },
            'connections': node.connections
        }
    
    def _connection_to_dict(self, connection: FlowchartConnection) -> Dict[str, Any]:
        """将连接转换为字典表示"""
        return {
            'from': connection.from_node,
            'to': connection.to_node,
            'label': connection.label
        } 