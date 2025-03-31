import os
import sys
from dotenv import load_dotenv
from modified_app import app
import setup_database

# 加载环境变量
load_dotenv()

# 检查是否安装了pymysql
try:
    import pymysql
    print("pymysql已安装，版本:", pymysql.__version__)
except ImportError:
    print("错误: 未检测到pymysql。请使用以下命令安装:")
    print("pip install pymysql")
    sys.exit(1)

if __name__ == '__main__':
    # 设置数据库
    print("正在设置MySQL数据库...")
    setup_database.setup_database()
    
    # 获取环境变量，如果不存在则使用默认值
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'
    
    print(f"启动服务器: http://{host}:{port}")
    print("按Ctrl+C停止服务器")
    
    # 启动Flask应用
    app.run(host=host, port=port, debug=debug)                                                                          