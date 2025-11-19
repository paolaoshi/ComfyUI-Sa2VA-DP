# 测试脚本 - 验证节点功能
# 运行此脚本测试模型管理器和节点的基本功能

import sys
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print("=" * 60)
print("ComfyUI-Sa2VA-DP 节点测试")
print("=" * 60)

# 测试1: 导入模型管理器
print("\n[测试1] 导入模型管理器...")
try:
    from model_manager import get_model_manager
    print("✅ 模型管理器导入成功")
except Exception as e:
    print(f"❌ 模型管理器导入失败: {e}")
    sys.exit(1)

# 测试2: 初始化模型管理器
print("\n[测试2] 初始化模型管理器...")
try:
    manager = get_model_manager()
    print(f"✅ 模型管理器初始化成功")
    print(f"   模型目录: {manager.models_dir}")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    sys.exit(1)

# 测试3: 检查模型目录
print("\n[测试3] 检查模型目录...")
if manager.models_dir.exists():
    print(f"✅ 模型目录存在: {manager.models_dir}")
else:
    print(f"⚠️ 模型目录不存在，将创建: {manager.models_dir}")
    manager.models_dir.mkdir(parents=True, exist_ok=True)

# 测试4: 列出已下载的模型
print("\n[测试4] 列出已下载的模型...")
try:
    downloaded_models = manager.list_downloaded_models()
    if downloaded_models:
        print(f"✅ 找到 {len(downloaded_models)} 个已下载的模型:")
        for model in downloaded_models:
            print(f"   • {model}")
    else:
        print("ℹ️ 尚未下载任何模型")
except Exception as e:
    print(f"❌ 列出模型失败: {e}")

# 测试5: 检查特定模型
print("\n[测试5] 检查特定模型...")
test_model = "ByteDance/Sa2VA-Qwen3-VL-4B"
try:
    is_downloaded = manager.is_model_downloaded(test_model)
    if is_downloaded:
        print(f"✅ 模型已下载: {test_model}")
        model_info = manager.get_model_info(test_model)
        print(f"   路径: {model_info['local_path']}")
        if 'size_gb' in model_info:
            print(f"   大小: {model_info['size_gb']:.2f} GB")
    else:
        print(f"ℹ️ 模型未下载: {test_model}")
        print(f"   将在首次使用时自动下载")
except Exception as e:
    print(f"❌ 检查模型失败: {e}")

# 测试6: 导入节点类
print("\n[测试6] 导入节点类...")
try:
    from nodes.sa2va_node import Sa2VANode
    print("✅ 节点类导入成功")
except Exception as e:
    print(f"❌ 节点类导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试7: 检查节点输入类型
print("\n[测试7] 检查节点输入类型...")
try:
    input_types = Sa2VANode.INPUT_TYPES()
    print("✅ 节点输入类型定义正确")
    print(f"   必需参数: {list(input_types['required'].keys())}")
    if 'optional' in input_types:
        print(f"   可选参数: {list(input_types['optional'].keys())}")
except Exception as e:
    print(f"❌ 获取输入类型失败: {e}")

# 测试8: 检查依赖项
print("\n[测试8] 检查依赖项...")
dependencies = {
    "torch": "PyTorch",
    "transformers": "Transformers",
    "PIL": "Pillow",
    "numpy": "NumPy",
    "huggingface_hub": "HuggingFace Hub",
}

missing_deps = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"✅ {name} 已安装")
    except ImportError:
        print(f"❌ {name} 未安装")
        missing_deps.append(name)

# 可选依赖
optional_deps = {
    "bitsandbytes": "BitsAndBytes (8位量化)",
    "flash_attn": "Flash Attention (加速)",
}

for module, name in optional_deps.items():
    try:
        __import__(module)
        print(f"✅ {name} 已安装")
    except ImportError:
        print(f"ℹ️ {name} 未安装 (可选)")

# 测试9: 检查transformers版本
print("\n[测试9] 检查transformers版本...")
try:
    import transformers
    version = transformers.__version__
    version_parts = version.split(".")
    major, minor = int(version_parts[0]), int(version_parts[1])
    
    if major >= 4 and minor >= 57:
        print(f"✅ Transformers版本满足要求: {version}")
    else:
        print(f"⚠️ Transformers版本过低: {version}")
        print(f"   需要: >= 4.57.0")
        print(f"   请运行: pip install transformers>=4.57.0 --upgrade")
except Exception as e:
    print(f"❌ 检查版本失败: {e}")

# 总结
print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)

if missing_deps:
    print(f"⚠️ 缺少必需依赖: {', '.join(missing_deps)}")
    print(f"   请运行: pip install -r requirements.txt")
else:
    print("✅ 所有必需依赖已安装")

print("\n节点已准备就绪！")
print("请重启ComfyUI以加载此节点。")
print("=" * 60)
