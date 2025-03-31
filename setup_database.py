#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据库初始化脚本
用于创建MySQL数据库和所需表
"""

import os
import datetime
import pymysql
from pymysql.cursors import DictCursor

def setup_database():
    """设置MySQL数据库和必要的表"""
    # MySQL连接配置
    config = {
        'host': 'localhost',
        'user': 'root',
        'password': '123456',
        'charset': 'utf8mb4',
        'cursorclass': DictCursor
    }
    
    db_name = 'flowchat_db'
    
    try:
        # 首先连接MySQL服务器（不指定数据库）
        conn = pymysql.connect(**config)
        cursor = conn.cursor()
        
        # 检查数据库是否存在，不存在则创建
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        print(f"确保数据库 {db_name} 存在")
        
        # 关闭当前连接
        conn.close()
        
        # 重新连接，这次连接到新创建的数据库
        config['db'] = db_name
        conn = pymysql.connect(**config)
        cursor = conn.cursor()
        
        # 检查表是否存在
        cursor.execute("SHOW TABLES LIKE 'analysis_history'")
        table_exists = cursor.fetchone()
        
        # 创建分析历史表
        if not table_exists:
            print("创建analysis_history表...")
            cursor.execute('''
            CREATE TABLE analysis_history (
                id VARCHAR(255) PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                image_path VARCHAR(255) NOT NULL,
                result_json LONGTEXT NOT NULL,
                date DATE NOT NULL,
                score INT,
                comment TEXT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            ''')
            
            # 确保上传目录存在
            os.makedirs('app/static/uploads', exist_ok=True)
            
            # 添加示例记录
            print("添加示例数据...")
            sample_data = [
                {
                    'id': 'sample-001',
                    'filename': '示例流程图1.png',
                    'image_path': 'uploads/sample1.png',
                    'result_json': '''
                    {
                        "shapes": [
                            {"type": "rectangle", "position": {"x": 100, "y": 100}, "center": [100, 100], "bbox": [80, 80, 40, 40], "text": "教师讲解", "area": 1600},
                            {"type": "diamond", "position": {"x": 200, "y": 200}, "center": [200, 200], "bbox": [180, 180, 40, 40], "text": "是否理解?", "area": 1600},
                            {"type": "ellipse", "position": {"x": 50, "y": 50}, "center": [50, 50], "bbox": [30, 30, 40, 40], "text": "开始", "area": 1600},
                            {"type": "parallelogram", "position": {"x": 300, "y": 300}, "center": [300, 300], "bbox": [280, 280, 40, 40], "text": "学生提交作业", "area": 1600}
                        ],
                        "text_regions": [
                            {"text": "教师讲解", "position": {"x": 100, "y": 100}, "center": [100, 100], "bbox": [80, 80, 40, 40], "confidence": 0.9},
                            {"text": "是否理解?", "position": {"x": 200, "y": 200}, "center": [200, 200], "bbox": [180, 180, 40, 40], "confidence": 0.9},
                            {"text": "开始", "position": {"x": 50, "y": 50}, "center": [50, 50], "bbox": [30, 30, 40, 40], "confidence": 0.9},
                            {"text": "学生提交作业", "position": {"x": 300, "y": 300}, "center": [300, 300], "bbox": [280, 280, 40, 40], "confidence": 0.9}
                        ],
                        "nodes": [
                            {"id": 0, "type": "start", "text": "开始", "position": {"x": 50, "y": 50}, "connections": [1]},
                            {"id": 1, "type": "activity", "text": "教师讲解", "position": {"x": 100, "y": 100}, "connections": [2]},
                            {"id": 2, "type": "decision", "text": "是否理解?", "position": {"x": 200, "y": 200}, "connections": [3, 1]},
                            {"id": 3, "type": "input_output", "text": "学生提交作业", "position": {"x": 300, "y": 300}, "connections": [4]},
                            {"id": 4, "type": "end", "text": "结束", "position": {"x": 350, "y": 350}, "connections": []}
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
                        },
                        "scores": {
                            "逻辑性": 22,
                            "合理性": 20,
                            "清晰度": 18,
                            "创新性": 15,
                            "总分": 75
                        },
                        "评分理由": {
                            "逻辑性": "流程图的步骤顺序符合教学逻辑，从开始到结束有明确的流向。教师讲解后通过判断学生是否理解来决定是否继续讲解或转向学生作业提交，逻辑结构清晰。",
                            "合理性": "区分了教师活动（讲解）和学生活动（提交作业），活动安排符合教学实际。判断点设置合理，根据学生理解情况进行教学调整。",
                            "清晰度": "使用简练语言描述活动，表达相对清晰，但部分活动描述可以更加具体，如可以说明教师讲解的具体内容或方式。",
                            "创新性": "流程图结构基本遵循传统教学模式，缺乏创新教学理念和技术的融入，如小组协作、技术辅助等现代教学元素。"
                        },
                        "feedback": "这是一个基本合格的教学流程图，逻辑性和合理性较好，教学步骤清晰，但缺乏创新性。流程图反映了传统的讲授式教学模式，可以考虑融入更多互动和现代教学元素。",
                        "改进建议": [
                            "在教师讲解环节中，可以添加使用多媒体或其他辅助工具的说明，提升教学效果",
                            "考虑增加学生互动或小组活动环节，促进学生参与度",
                            "可以在学生提交作业后增加教师评价和反馈环节，形成闭环",
                            "为流程图添加时间分配指示，使教学计划更加明确"
                        ],
                        "符合规范": true,
                        "validation": {
                            "score": 85,
                            "feedback": "整体符合教学流程图规范，图形选择和文本描述基本合理。",
                            "details": [
                                {
                                    "shape_type": "ellipse",
                                    "text": "开始",
                                    "is_valid": true,
                                    "reason": "椭圆形正确用于表示流程的起始点"
                                },
                                {
                                    "shape_type": "rectangle",
                                    "text": "教师讲解",
                                    "is_valid": true,
                                    "reason": "矩形正确用于表示教学活动步骤"
                                },
                                {
                                    "shape_type": "diamond",
                                    "text": "是否理解?",
                                    "is_valid": true,
                                    "reason": "菱形正确用于表示决策点，问题描述清晰"
                                },
                                {
                                    "shape_type": "parallelogram",
                                    "text": "学生提交作业",
                                    "is_valid": true,
                                    "reason": "平行四边形用于表示输入/输出，此处表示学生作业输出"
                                }
                            ]
                        },
                        "module_check": {
                            "has_valid_modules": true,
                            "modules_count": 1,
                            "modules": [[1, 3, 4]],
                            "message": "检测到有效的流程图模块组合，教学设计结构完整"
                        }
                    }
                    ''',
                    'date': datetime.datetime.now().strftime('%Y-%m-%d'),
                    'score': 75,
                    'comment': "基本合格的教学流程图，逻辑性和合理性较好，但创新性不足"
                },
                {
                    'id': 'sample-002',
                    'filename': '示例流程图2.png',
                    'image_path': 'uploads/sample2.png',
                    'result_json': '''
                    {
                        "shapes": [
                            {"type": "rectangle", "position": {"x": 100, "y": 100}, "center": [100, 100], "bbox": [80, 80, 40, 40], "text": "教师引入课题", "area": 1600},
                            {"type": "rectangle", "position": {"x": 150, "y": 150}, "center": [150, 150], "bbox": [130, 130, 40, 40], "text": "小组讨论", "area": 1600},
                            {"type": "diamond", "position": {"x": 200, "y": 200}, "center": [200, 200], "bbox": [180, 180, 40, 40], "text": "是否达成共识?", "area": 1600},
                            {"type": "ellipse", "position": {"x": 50, "y": 50}, "center": [50, 50], "bbox": [30, 30, 40, 40], "text": "开始", "area": 1600},
                            {"type": "parallelogram", "position": {"x": 300, "y": 300}, "center": [300, 300], "bbox": [280, 280, 40, 40], "text": "学生展示成果", "area": 1600},
                            {"type": "rectangle", "position": {"x": 350, "y": 350}, "center": [350, 350], "bbox": [330, 330, 40, 40], "text": "教师总结提升", "area": 1600},
                            {"type": "ellipse", "position": {"x": 400, "y": 400}, "center": [400, 400], "bbox": [380, 380, 40, 40], "text": "结束", "area": 1600}
                        ],
                        "text_regions": [
                            {"text": "教师引入课题", "position": {"x": 100, "y": 100}, "center": [100, 100], "bbox": [80, 80, 40, 40], "confidence": 0.95},
                            {"text": "小组讨论", "position": {"x": 150, "y": 150}, "center": [150, 150], "bbox": [130, 130, 40, 40], "confidence": 0.93},
                            {"text": "是否达成共识?", "position": {"x": 200, "y": 200}, "center": [200, 200], "bbox": [180, 180, 40, 40], "confidence": 0.9},
                            {"text": "开始", "position": {"x": 50, "y": 50}, "center": [50, 50], "bbox": [30, 30, 40, 40], "confidence": 0.98},
                            {"text": "学生展示成果", "position": {"x": 300, "y": 300}, "center": [300, 300], "bbox": [280, 280, 40, 40], "confidence": 0.92},
                            {"text": "教师总结提升", "position": {"x": 350, "y": 350}, "center": [350, 350], "bbox": [330, 330, 40, 40], "confidence": 0.94},
                            {"text": "结束", "position": {"x": 400, "y": 400}, "center": [400, 400], "bbox": [380, 380, 40, 40], "confidence": 0.97}
                        ],
                        "nodes": [
                            {"id": 0, "type": "start", "text": "开始", "position": {"x": 50, "y": 50}, "connections": [1]},
                            {"id": 1, "type": "activity", "text": "教师引入课题", "position": {"x": 100, "y": 100}, "connections": [2]},
                            {"id": 2, "type": "activity", "text": "小组讨论", "position": {"x": 150, "y": 150}, "connections": [3]},
                            {"id": 3, "type": "decision", "text": "是否达成共识?", "position": {"x": 200, "y": 200}, "connections": [4, 2]},
                            {"id": 4, "type": "input_output", "text": "学生展示成果", "position": {"x": 300, "y": 300}, "connections": [5]},
                            {"id": 5, "type": "activity", "text": "教师总结提升", "position": {"x": 350, "y": 350}, "connections": [6]},
                            {"id": 6, "type": "end", "text": "结束", "position": {"x": 400, "y": 400}, "connections": []}
                        ],
                        "statistics": {
                            "total_nodes": 7,
                            "node_types": {
                                "activity": 3,
                                "decision": 1,
                                "start": 1,
                                "end": 1,
                                "input_output": 1
                            }
                        },
                        "scores": {
                            "逻辑性": 24,
                            "合理性": 23,
                            "清晰度": 22,
                            "创新性": 21,
                            "总分": 90
                        },
                        "评分理由": {
                            "逻辑性": "流程图结构完整，教学环节连贯，从引入到讨论、展示和总结，符合教学活动的逻辑顺序。判断点设置合理，在小组讨论后判断是否达成共识。",
                            "合理性": "教师活动和学生活动区分明确，既有教师主导环节，也有学生主动参与环节。小组讨论和成果展示促进了学生间的协作学习。",
                            "清晰度": "各环节描述简练明确，活动内容一目了然。判断条件表述清晰，分支走向明确。",
                            "创新性": "融入了小组合作学习模式，注重学生参与和互动，体现了现代教学理念。学生展示环节促进了学习成果的分享和交流。"
                        },
                        "feedback": "这是一个优秀的教学流程图，体现了"以学生为中心"的教学理念。流程图逻辑严密，环节设计合理，既重视学生主动参与，又有教师适当引导。小组讨论和成果展示环节促进了学生的深度学习和协作能力培养。教师在开始和结束阶段的引导和总结，确保了教学目标的达成。",
                        "改进建议": [
                            "可以考虑在小组讨论前增加学习任务说明或提供讨论指导",
                            "学生展示成果环节可以补充同伴评价机制，提高参与度",
                            "教师总结环节可以增加学生反思或拓展思考的引导",
                            "可以尝试添加更多的技术支持手段，如数字工具的应用"
                        ],
                        "符合规范": true,
                        "validation": {
                            "score": 95,
                            "feedback": "流程图设计规范，图形选择和文本描述高度合理，符合教学流程图标准。",
                            "details": [
                                {
                                    "shape_type": "ellipse",
                                    "text": "开始",
                                    "is_valid": true,
                                    "reason": "椭圆形正确用于表示流程的起始点"
                                },
                                {
                                    "shape_type": "rectangle",
                                    "text": "教师引入课题",
                                    "is_valid": true,
                                    "reason": "矩形正确用于表示教学活动步骤"
                                },
                                {
                                    "shape_type": "rectangle",
                                    "text": "小组讨论",
                                    "is_valid": true,
                                    "reason": "矩形正确用于表示教学活动步骤"
                                },
                                {
                                    "shape_type": "diamond",
                                    "text": "是否达成共识?",
                                    "is_valid": true,
                                    "reason": "菱形正确用于表示决策点，问题描述清晰"
                                },
                                {
                                    "shape_type": "parallelogram",
                                    "text": "学生展示成果",
                                    "is_valid": true,
                                    "reason": "平行四边形用于表示输入/输出，此处表示学生成果展示"
                                },
                                {
                                    "shape_type": "rectangle",
                                    "text": "教师总结提升",
                                    "is_valid": true,
                                    "reason": "矩形正确用于表示教学活动步骤"
                                },
                                {
                                    "shape_type": "ellipse",
                                    "text": "结束",
                                    "is_valid": true,
                                    "reason": "椭圆形正确用于表示流程的结束点"
                                }
                            ]
                        },
                        "module_check": {
                            "has_valid_modules": true,
                            "modules_count": 2,
                            "modules": [[1, 4, 5], [2, 4, 5]],
                            "message": "检测到多个有效的流程图模块组合，教学设计结构完整且丰富"
                        }
                    }
                    ''',
                    'date': (datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d'),
                    'score': 90,
                    'comment': "优秀的教学流程图，体现了'以学生为中心'的教学理念"
                }
            ]
            
            for data in sample_data:
                cursor.execute(
                    """
                    INSERT INTO analysis_history (id, filename, image_path, result_json, date, score, comment)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (data['id'], data['filename'], data['image_path'], data['result_json'], data['date'], data['score'], data['comment'])
                )
        
        # 提交更改并关闭连接
        conn.commit()
        conn.close()
        
        print(f"MySQL数据库设置完成: {db_name}")
        
    except Exception as e:
        print(f"设置数据库时出错: {e}")
        raise

if __name__ == "__main__":
    setup_database() 