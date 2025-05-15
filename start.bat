@echo off
REM === 启动脚本：激活 Conda 环境并运行 Python 项目 ===

REM 设置环境名称
set ENV_NAME=Final

REM 检查 Anaconda 是否安装
where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Please First Install Anaconda or Miniconda。
    pause
    exit /b
)

REM 初始化 Conda（重要：激活环境前必须初始化）
call "%USERPROFILE%\anaconda3\Scripts\activate.bat"

REM 激活环境
call conda activate %ENV_NAME%

REM 运行主程序（替换为你要启动的脚本）
python app.py

REM 如果想运行其他脚本也可以在这里追加
REM python run_flask.py
REM python detect_sound.py

pause
