@echo off
chcp 65001 >nul
echo ========================================
echo ComfyUI-Sa2VA-DP 安装验证
echo ========================================
echo.

echo 正在运行测试脚本...
echo.

E:\Comfyui_test\python_dapao311\python.exe test_node.py

echo.
echo ========================================
echo 验证完成！
echo ========================================
echo.
echo 如果所有测试都通过，请重启ComfyUI。
echo.
pause
