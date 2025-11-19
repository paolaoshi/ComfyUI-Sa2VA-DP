# 自动安装脚本
# ComfyUI会在加载节点时自动运行这个脚本

import subprocess
import sys
import os
from pathlib import Path


def install_dependencies():
    """安装节点所需的依赖项"""
    
    print("=" * 60)
    print("ComfyUI-Sa2VA-DP 节点安装")
    print("=" * 60)
    
    # 获取requirements.txt路径
    current_dir = Path(__file__).parent
    requirements_file = current_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ 找不到requirements.txt文件")
        return False
    
    print(f"📦 开始安装依赖项...")
    print(f"📁 Requirements文件: {requirements_file}")
    
    try:
        # 使用pip安装依赖
        # 注意：这里使用sys.executable确保使用正确的Python环境
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(requirements_file),
                "--upgrade",
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10分钟超时
        )
        
        if result.returncode == 0:
            print("✅ 依赖项安装成功")
            print("\n" + "=" * 60)
            print("安装完成！")
            print("=" * 60)
            print("\n可选优化（需要CUDA）：")
            print("  • 8位量化: pip install bitsandbytes")
            print("  • Flash Attention: pip install flash-attn")
            print("\n模型将自动下载到: ComfyUI/models/Sa2VA/")
            print("=" * 60)
            return True
        else:
            print(f"❌ 安装失败")
            print(f"错误信息: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 安装超时（超过10分钟）")
        return False
    except Exception as e:
        print(f"❌ 安装过程中出错: {e}")
        return False


# 运行安装
if __name__ == "__main__":
    install_dependencies()
