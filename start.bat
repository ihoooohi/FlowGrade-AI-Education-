@echo off
echo 正在检查python环境...
python --version
if %errorlevel% neq 0 (
    echo 错误: 未检测到Python，请先安装Python 3.8或更高版本
    pause
    exit /b
)

echo 正在检查MySQL服务...
sc query mysql
if %errorlevel% neq 0 (
    echo 警告: 未检测到MySQL服务，请确保MySQL已安装并启动
    echo 如果您已安装MySQL，请手动启动服务
    echo 如果未安装，请先安装MySQL 5.7或更高版本
    pause
)

echo 是否要创建虚拟环境并安装依赖？(Y/N)
set /p choice=
if /i "%choice%"=="Y" (
    echo 创建虚拟环境...
    python -m venv flowchart_env
    
    echo 激活虚拟环境...
    call flowchart_env\Scripts\activate
    
    echo 安装依赖...
    pip install -r requirements.txt
    
    echo 依赖安装完成！
) else (
    echo 跳过依赖安装，确保已经安装了所有需要的包
)

echo 是否立即启动应用？(Y/N)
set /p start=
if /i "%start%"=="Y" (
    echo 启动流程图智能批阅系统...
    if exist flowchart_env (
        call flowchart_env\Scripts\activate
    )
    python run.py
) else (
    echo 您可以稍后通过运行 "python run.py" 启动应用
)

pause 