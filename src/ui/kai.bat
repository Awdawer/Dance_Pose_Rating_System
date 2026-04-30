@echo off
setlocal enabledelayedexpansion

:: 定义目标文件名
set "TARGET_FILE=gui_app.py"

:: 1. 先检查当前目录
if exist "%TARGET_FILE%" (
    echo 找到文件：%cd%\%TARGET_FILE%
    python "%TARGET_FILE%"
    pause
    exit /b
)

:: 2. 向上查找一级目录
cd ..
if exist "%TARGET_FILE%" (
    echo 找到文件：%cd%\%TARGET_FILE%
    python "%TARGET_FILE%"
    pause
    exit /b
)

:: 3. 向上查找两级目录
cd ..
if exist "%TARGET_FILE%" (
    echo 找到文件：%cd%\%TARGET_FILE%
    python "%TARGET_FILE%"
    pause
    exit /b
)

:: 4. 如果都没找到，提示错误
echo 错误：找不到 %TARGET_FILE%
echo 请确保此脚本放在项目目录中
pause