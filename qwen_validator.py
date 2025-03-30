import json
import requests

class QwenValidator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        
    def validate_shape_text(self, nodes):
        """使用qwen验证图形和文本的匹配"""
        print("开始验证图形和文本匹配...")
        print(f"节点数量: {len(nodes)}")
        
        # 检查节点数据
        if not nodes:
            print("警告: 节点数据为空")
            return {
                "success": False,
                "error": "节点数据为空"
            }
        
        # 检查节点数据格式
        for i, node in enumerate(nodes):
            if 'shape_type' not in node or 'text' not in node:
                print(f"警告: 节点 {i+1} 缺少shape_type或text字段")
                return {
                    "success": False,
                    "error": f"节点 {i+1} 数据格式错误，缺少必要字段"
                }
        
        # 构建提示词
        prompt = self._build_prompt(nodes)
        print("生成的提示词:")
        print(prompt)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 使用兼容模式的messages格式
        data = {
            "model": "qwen-plus",
            "messages": [
                {"role": "system", "content": "你是一个专业的流程图符号验证专家，负责评估流程图中的图形与文本是否匹配。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "top_p": 0.8,
            "max_tokens": 2000
        }
        
        try:
            print("开始调用千问API...")
            response = requests.post(self.api_url, headers=headers, json=data)
            print(f"API响应状态码: {response.status_code}")
            
            if response.status_code != 200:
                print(f"API错误响应: {response.text}")
                
            response.raise_for_status()
            result = response.json()
            print("成功获取API响应")
            
            parse_result = self._parse_response(result)
            print(f"解析结果: {parse_result['success']}")
            return parse_result
        except requests.RequestException as e:
            error_msg = f"API请求错误: {str(e)}"
            print(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"调用千问API时发生未知错误: {str(e)}"
            print(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
    
    def _build_prompt(self, nodes):
        """构建提示词"""
        rules = """请依据以下规则判断每个图形与文本的匹配是否正确：
1. 矩形框：必须表示教学活动或步骤（例如"教师讲解"）
2. 平行四边形：必须表示输入/输出（例如"学生提交作业"）
3. 菱形框：必须表示决策或分支（例如"是否达标？"）
4. 圆角矩形：在流程开头和结尾表示流程的开始和结束（例如"开始"、"结束"），在流程中间出现时表示教学媒介和演示内容（例如"多媒体课件展示"）

对于每个图形，请判断：
1. 图形类型与文本内容是否匹配
2. 给出具体的判断理由
3. 如果不匹配，请给出修改建议

请以JSON格式返回结果，格式如下：
{
    "overall_valid": true/false,
    "score": 0-100的整数,
    "feedback": "总体评价",
    "details": [
        {
            "shape_type": "图形类型",
            "text": "文本内容",
            "is_valid": true/false,
            "reason": "判断理由",
            "suggestion": "修改建议（如果需要）"
        }
    ]
}
"""
        shapes_text = "以下是需要验证的图形和文本：\n"
        for i, node in enumerate(nodes, 1):
            shapes_text += f"{i}. 图形类型：{node['shape_type']}, 文本内容：{node['text']}\n"
            
        return f"{rules}\n\n{shapes_text}"
    
    def _parse_response(self, response):
        """解析qwen的响应"""
        try:
            print("开始解析API响应...")
            print(f"响应结构: {response.keys()}")
            
            # 兼容模式的返回格式
            if 'choices' not in response:
                print(f"响应格式错误，缺少choices字段: {response}")
                return {
                    "success": False,
                    "error": "API响应格式错误，缺少choices字段"
                }
            
            if not response['choices'] or 'message' not in response['choices'][0]:
                print(f"响应格式错误，缺少message字段: {response['choices']}")
                return {
                    "success": False,
                    "error": "API响应格式错误，缺少message字段"
                }
                
            if 'content' not in response['choices'][0]['message']:
                print(f"响应格式错误，缺少content字段: {response['choices'][0]['message']}")
                return {
                    "success": False,
                    "error": "API响应格式错误，缺少content字段"
                }
                
            result_text = response['choices'][0]['message']['content']
            print(f"获取到响应文本: {result_text[:100]}...")  # 只打印前100个字符
            
            # 识别JSON格式的内容
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                print(f"响应文本中未找到JSON: {result_text}")
                return {
                    "success": False,
                    "error": "响应文本中未找到有效的JSON"
                }
                
            json_text = result_text[json_start:json_end]
            # 解析JSON
            result = json.loads(json_text)
            print("成功解析JSON")
            
            # 验证结果结构
            required_fields = ["overall_valid", "score", "feedback", "details"]
            for field in required_fields:
                if field not in result:
                    print(f"结果缺少必要字段 {field}")
                    return {
                        "success": False,
                        "error": f"结果格式错误，缺少{field}字段"
                    }
            
            return {
                "success": True,
                "result": result
            }
        except json.JSONDecodeError as e:
            error_msg = f"JSON解析失败: {str(e)}, 原始文本: {result_text[:200]}..."
            print(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            import traceback
            error_msg = f"解析响应失败: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return {
                "success": False,
                "error": error_msg
            } 