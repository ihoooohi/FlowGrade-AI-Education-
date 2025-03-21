import os
import json
import base64
from typing import Dict, Any, List
import requests
from openai import OpenAI
import cv2

class AIEvaluator:
    """使用qwen2.5-vl模型评估流程图"""
    
    def __init__(self):
        """初始化AI评估器"""
        # 从环境变量获取API密钥
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未设置DASHSCOPE_API_KEY环境变量")
        
        # 初始化OpenAI客户端（使用百炼API）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 评估规则
        self.evaluation_rules = {
            "逻辑性": "步骤顺序要符合教学逻辑（如'导入→讲授→练习→总结'）",
            "合理性": "能区分教师活动（如'讲解案例'）与学生活动（如'分组实验'）",
            "清晰度": "能用简练语言描述活动（如'案例导入'而非'教师使用案例导入新课'）",
            "创新性": "是否融入非传统技术或教学模式（如翻转课堂、项目式学习）"
        }
        
        # 标准符号规则
        self.symbol_rules = {
            "矩形框": "表示教学活动或步骤（如'教师讲解'）",
            "平行四边形": "表示输入/输出（如'学生提交作业'）",
            "菱形框": "表示决策或分支（如'是否达标？'）",
            "箭头": "表示流程方向",
            "椭圆形": "表示起点或终点（如'开始''结束'）"
        }
        
        # 基本规范
        self.basic_rules = {
            "条件判断": "分支流程要有明确的条件判断（如'是否完成任务→是/否'）",
            "时间标注": "在步骤旁标注预计时间分配（如'小组讨论：5分钟'）",
            "流程方向": "自上而下，不能交叉连线",
            "基本规范": "一个矩形、平行四边形、圆角矩形（中间有竖线）三者成模块出现"
        }
    
    def evaluate(self, structured_data: Dict[str, Any], image_path: str) -> Dict[str, Any]:
        """
        评估流程图
        
        Args:
            structured_data: 流程图的结构化表示
            image_path: 流程图图像路径
            
        Returns:
            评估结果
        """
        # 准备评估提示
        prompt = self._prepare_evaluation_prompt(structured_data)
        
        # 读取图像并转换为base64
        image_base64 = self._encode_image(image_path)
        
        try:
            # 调用AI模型进行评估
            response = self.client.chat.completions.create(
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
            return {
                "error": str(e),
                "scores": {},
                "feedback": "AI评估失败，请稍后重试。",
                "statistics": structured_data.get("statistics", {})
            }
    
    def _prepare_evaluation_prompt(self, structured_data: Dict[str, Any]) -> str:
        """准备评估提示"""
        # 将结构化数据转换为JSON字符串
        data_json = json.dumps(structured_data, ensure_ascii=False, indent=2)
        
        # 构建提示
        prompt = f"""你是一个教学流程图评估专家。请根据以下规则评估这张教学流程图：

## 标准符号规则
- 矩形框：表示教学活动或步骤（如"教师讲解"）
- 平行四边形：表示输入/输出（如"学生提交作业"）
- 菱形框：表示决策或分支（如"是否达标？"）
- 箭头：表示流程方向
- 椭圆形：表示起点或终点（如"开始""结束"）
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
{data_json}
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
    "逻辑性": 分数,
    "合理性": 分数,
    "清晰度": 分数,
    "创新性": 分数,
    "总分": 分数
  },
  "feedback": "详细的反馈意见",
  "符合规范": true/false,
  "改进建议": ["建议1", "建议2", ...]
}
```"""
        
        return prompt
    
    def _encode_image(self, image_path: str) -> str:
        """将图像编码为base64字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _parse_evaluation_result(self, response_text: str) -> Dict[str, Any]:
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