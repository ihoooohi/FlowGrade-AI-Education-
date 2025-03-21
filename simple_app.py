import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

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
        
        # 模拟分析结果
        mock_result = {
            "scores": {
                "逻辑性": 8,
                "合理性": 7,
                "清晰度": 9,
                "创新性": 6,
                "总分": 75
            },
            "feedback": "这是一个示例反馈。在实际系统中，这里会显示基于AI分析的详细反馈。",
            "符合规范": True,
            "改进建议": [
                "建议1：可以考虑添加更多的决策点，使流程更加灵活。",
                "建议2：教师活动和学生活动的区分可以更加明确。",
                "建议3：可以考虑添加时间标注，以便更好地控制教学节奏。"
            ],
            "statistics": {
                "total_nodes": 10,
                "node_types": {
                    "activity": 5,
                    "decision": 2,
                    "start": 1,
                    "end": 1,
                    "input_output": 1
                }
            }
        }
        
        # 返回处理结果
        return render_template('result.html', 
                              image_path=f'uploads/{filename}',
                              result=mock_result)
    
    flash('不支持的文件类型')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True) 