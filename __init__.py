# ComfyUI-Sa2VA-DP - Sa2VA 分割节点 for ComfyUI
# 改进版本：自动模型下载和管理

__version__ = "1.0.3"

from .nodes.sa2va_node import Sa2VANode

# 定义节点类映射
NODE_CLASS_MAPPINGS = {
    "Sa2VANode": Sa2VANode,
}

# 定义节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "Sa2VANode": "Sa2VA 图像分割@炮老师的小课堂",
}

# 暴露映射给ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("✅ ComfyUI-Sa2VA-DP 节点已加载")
