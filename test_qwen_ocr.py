import os
import cv2
import numpy as np
import sys
import json
from dotenv import load_dotenv
from modified_app import QwenVLOCREngine

# 设置控制台编码
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 加载环境变量
load_dotenv(override=True)

def create_test_flowchart(image_path):
    """创建一个测试流程图"""
    # 创建一个空白图像
    img = 255 * np.ones((600, 800, 3), dtype=np.uint8)
    
    # 绘制流程图元素
    # 1. 开始节点 (椭圆)
    cv2.ellipse(img, (400, 50), (80, 30), 0, 0, 360, (0, 0, 0), 2)
    cv2.putText(img, "Start", (375, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 箭头
    cv2.arrowedLine(img, (400, 80), (400, 120), (0, 0, 0), 2, tipLength=0.1)
    
    # 2. 输入节点 (平行四边形)
    pts = np.array([[320, 120], [480, 120], [450, 170], [290, 170]], np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 0), 2)
    cv2.putText(img, "Input Data", (350, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 箭头
    cv2.arrowedLine(img, (400, 170), (400, 220), (0, 0, 0), 2, tipLength=0.1)
    
    # 3. 处理节点 (矩形)
    cv2.rectangle(img, (300, 220), (500, 270), (0, 0, 0), 2)
    cv2.putText(img, "Process Data", (340, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 箭头
    cv2.arrowedLine(img, (400, 270), (400, 320), (0, 0, 0), 2, tipLength=0.1)
    
    # 4. 决策节点 (菱形)
    pts = np.array([[400, 320], [500, 370], [400, 420], [300, 370]], np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 0), 2)
    cv2.putText(img, "Is Valid?", (360, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 箭头 - 是
    cv2.arrowedLine(img, (400, 420), (400, 470), (0, 0, 0), 2, tipLength=0.1)
    cv2.putText(img, "Yes", (410, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 箭头 - 否
    cv2.arrowedLine(img, (300, 370), (200, 370), (0, 0, 0), 2, tipLength=0.1)
    cv2.putText(img, "No", (240, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 5. 处理节点 - 错误处理 (矩形)
    cv2.rectangle(img, (100, 340), (200, 400), (0, 0, 0), 2)
    cv2.putText(img, "Error", (120, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 箭头 - 返回处理
    cv2.arrowedLine(img, (150, 340), (150, 250), (0, 0, 0), 2, tipLength=0.1)
    cv2.arrowedLine(img, (150, 250), (300, 250), (0, 0, 0), 2, tipLength=0.1)
    
    # 6. 输出节点 (平行四边形)
    pts = np.array([[320, 470], [480, 470], [450, 520], [290, 520]], np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 0), 2)
    cv2.putText(img, "Output Result", (330, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 箭头
    cv2.arrowedLine(img, (400, 520), (400, 570), (0, 0, 0), 2, tipLength=0.1)
    
    # 7. 结束节点 (椭圆)
    cv2.ellipse(img, (400, 570), (80, 30), 0, 0, 360, (0, 0, 0), 2)
    cv2.putText(img, "End", (385, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 保存图像
    cv2.imwrite(image_path, img)
    print(f"已创建测试流程图: {image_path}")
    return img

def test_qwen_ocr():
    """测试千问视觉OCR功能"""
    print("开始测试千问视觉OCR功能...")
    
    # 初始化OCR引擎
    ocr_engine = QwenVLOCREngine()
    
    # 检查OCR引擎是否初始化成功
    if ocr_engine.client is None:
        print("千问视觉OCR客户端初始化失败")
        return
    
    # 测试图像路径
    test_image_path = "app/static/uploads/test_flowchart.jpg"
    
    # 如果测试图像不存在，创建一个测试流程图
    if not os.path.exists(test_image_path):
        print(f"测试图像不存在: {test_image_path}")
        # 确保目录存在
        os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
        # 创建测试流程图
        img = create_test_flowchart(test_image_path)
    else:
        # 读取测试图像
        img = cv2.imread(test_image_path)
        if img is None:
            print(f"无法读取测试图像: {test_image_path}")
            return
    
    # 提取文本
    print("开始提取文本...")
    text_regions = ocr_engine.extract_text(img)
    
    # 打印结果
    print(f"识别到 {len(text_regions)} 个文本区域:")
    for i, region in enumerate(text_regions):
        print(f"  {i+1}. 文本: {region['text']}")
        print(f"     置信度: {region.get('confidence', 'N/A')}")
        print(f"     位置: {region['bbox']}")
        print(f"     中心点: {region['center']}")
    
    # 将结果保存到文件
    result_file = "ocr_result.json"
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(text_regions, f, ensure_ascii=False, indent=2)
        print(f"OCR结果已保存到: {result_file}")
    except Exception as e:
        print(f"保存OCR结果时出错: {str(e)}")
    
    # 创建可视化结果
    vis_image = img.copy()
    for i, region in enumerate(text_regions):
        x, y, w, h = region["bbox"]
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(vis_image, f"{i+1}: {region['text']}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 保存可视化结果
    vis_path = "app/static/uploads/ocr_result_vis.jpg"
    cv2.imwrite(vis_path, vis_image)
    print(f"可视化结果已保存到: {vis_path}")
    
    print("测试完成")

if __name__ == "__main__":
    test_qwen_ocr() 