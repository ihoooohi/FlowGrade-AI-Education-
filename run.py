import os
from dotenv import load_dotenv
from modified_app import app

# 加载环境变量
load_dotenv()

if __name__ == '__main__':
    # 获取环境变量，如果不存在则使用默认值
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'
    
    print(f"启动服务器: http://{host}:{port}")
    print("按Ctrl+C停止服务器")
    
    # 启动Flask应用
    app.run(host=host, port=port, debug=debug)                                                                          