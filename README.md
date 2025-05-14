# 教学流程图智能批阅系统

这是一个基于人工智能技术的教学流程图智能批阅系统，可以自动分析和评估教学流程图，提供专业评价和改进建议。

## 系统要求

- Python 3.8+
- MySQL 5.7+
- 其他依赖见 requirements.txt

## 配置说明

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. MySQL数据库配置

如需修改数据库连接信息，请编辑以下文件：
- `setup_database.py` - 初始化数据库
- `modified_app.py` - `get_db_connection()` 函数

### 3. 启动系统

```bash
python run.py
```

系统会自动初始化数据库并启动web服务

## 主要功能

1. 智能识别和分析流程图中的各种元素
2. 评估流程图的逻辑性、合理性、清晰度和创新性
3. 提供专业的评价和改进建议
4. 支持历史记录管理和分析报告下载
5. 流程图规范验证和模块组合检查

## 文件结构

- `modified_app.py` - 主应用入口，包含所有路由和核心功能
- `setup_database.py` - 数据库初始化脚本
- `run.py` - 应用启动脚本
- `app/templates/` - HTML模板目录
- `app/static/` - 静态资源目录

## 注意事项

首次运行前请确保MySQL服务已启动，并且用户具有创建数据库的权限。

## 功能特点

- 图像处理：自动识别流程图中的各种图形（矩形、菱形、椭圆等）
- OCR文本识别：提取流程图中的文本内容
- 流程图分析：分析流程图的结构和逻辑关系
- AI评估：基于教学设计原则对流程图进行评估和打分
- 反馈建议：提供改进建议和详细反馈

## 安装指南

1. 克隆或下载本仓库
2. 创建并激活虚拟环境（推荐）：

```bash
python -m venv flowchart_env
# Windows
flowchart_env\Scripts\activate
# Linux/Mac
source flowchart_env/bin/activate
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

4. 配置环境变量：
   - 复制 `.env.example` 文件为 `.env`
   - 在 `.env` 文件中填入您的API密钥和其他配置

## 使用方法

1. 启动应用：

```bash
python run.py
```

2. 在浏览器中访问
3. 上传流程图图像进行分析和评估

## 模拟模式

如果某些依赖无法安装，系统会自动切换到模拟模式：

- 如果OpenCV不可用：使用模拟图像处理
- 如果PaddleOCR不可用：使用模拟OCR功能
- 如果OpenAI API不可用：使用模拟AI评估

模拟模式下，系统仍然可以运行，但会使用预设的示例数据而非实际分析结果。

## API接口

系统提供了REST API接口，可用于集成到其他应用中：

- `POST /api/analyze`：上传并分析流程图

## 功能展示
# 1. 上传界面
![Image](https://github.com/user-attachments/assets/8777bf02-22e6-4662-a406-17787e7614b9)
# 2. 分析结果
![Image](https://github.com/user-attachments/assets/5fb7d09c-8429-4a59-94e7-47ef19644bda)
![Image](https://github.com/user-attachments/assets/750124d4-f1d5-4d25-a106-1cdd58ce7631)
![Image](https://github.com/user-attachments/assets/08e18222-fa6d-426e-adf5-0c6705364fa2)
![Image](https://github.com/user-attachments/assets/0e01b6ee-eaf7-4cd9-81b4-94f9b5473bd8)
![Image](https://github.com/user-attachments/assets/79699d8e-9b90-4909-a66a-e5d197269a0e)
![Image](https://github.com/user-attachments/assets/5d2f1b13-c543-4d87-8474-bf769eb10ad6)
# 3. 分析历史
![Image](https://github.com/user-attachments/assets/5a1aa400-3188-4bf9-af57-1cff9ae8ad29)