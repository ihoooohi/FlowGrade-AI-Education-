# 教学流程图智能批阅系统

这是一个用于自动分析和评估教学流程图的系统。该系统能够处理上传的流程图图像，识别其中的图形和文本，分析流程图的结构，并提供智能评估和反馈。

## 功能特点

- 图像处理：自动识别流程图中的各种图形（矩形、菱形、椭圆等）
- OCR文本识别：提取流程图中的文本内容
- 流程图分析：分析流程图的结构和逻辑关系
- AI评估：基于教学设计原则对流程图进行评估和打分
- 反馈建议：提供改进建议和详细反馈

## 系统要求

- Python 3.8+
- 依赖库：详见 `requirements.txt`

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

2. 在浏览器中访问：`http://localhost:5000`
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

## 许可证

MIT 