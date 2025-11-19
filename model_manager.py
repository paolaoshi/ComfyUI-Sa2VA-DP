# Sa2VA 模型下载和管理器
# 负责自动下载模型到指定目录，并检测已存在的模型

import os
import torch
from pathlib import Path
from typing import Optional, Tuple


class Sa2VAModelManager:
    """Sa2VA模型管理器 - 处理模型下载和缓存"""
    
    def __init__(self, comfyui_path: str = "E:/Comfyui_test/ComfyUI"):
        """
        初始化模型管理器
        
        Args:
            comfyui_path: ComfyUI的根目录路径
        """
        self.comfyui_path = Path(comfyui_path)
        # 模型存储目录：ComfyUI/models/Sa2VA
        self.models_dir = self.comfyui_path / "models" / "Sa2VA"
        
        # 确保模型目录存在
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Sa2VA模型目录: {self.models_dir}")
    
    def get_model_path(self, model_name: str) -> Path:
        """
        获取模型的本地存储路径
        
        Args:
            model_name: 模型名称，例如 "ByteDance/Sa2VA-Qwen3-VL-4B"
        
        Returns:
            模型的本地路径
        """
        # 从完整名称中提取模型简称
        # 例如: "ByteDance/Sa2VA-Qwen3-VL-4B" -> "Sa2VA-Qwen3-VL-4B"
        model_short_name = model_name.split("/")[-1]
        return self.models_dir / model_short_name
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """
        检查模型是否已经下载
        
        Args:
            model_name: 模型名称
        
        Returns:
            True如果模型已下载，False否则
        """
        model_path = self.get_model_path(model_name)
        
        # 检查目录是否存在
        if not model_path.exists():
            return False
        
        # 检查关键文件是否存在
        # Sa2VA模型通常包含这些文件
        required_files = [
            "config.json",           # 模型配置
            "model.safetensors",     # 模型权重（safetensors格式）
        ]
        
        # 也可能是pytorch格式
        alternative_files = [
            "pytorch_model.bin",     # 模型权重（pytorch格式）
        ]
        
        # 检查是否有必需的配置文件
        has_config = (model_path / "config.json").exists()
        
        # 检查是否有模型权重文件（safetensors或pytorch格式）
        has_weights = (
            (model_path / "model.safetensors").exists() or
            (model_path / "pytorch_model.bin").exists() or
            any((model_path / f"model-{i:05d}-of-*.safetensors").exists() 
                for i in range(1, 100))  # 分片模型
        )
        
        if has_config and has_weights:
            print(f"✅ 检测到已下载的模型: {model_path}")
            return True
        
        return False
    
    def download_model(
        self, 
        model_name: str, 
        force_download: bool = False
    ) -> Tuple[bool, str]:
        """
        下载模型到本地目录
        
        Args:
            model_name: HuggingFace模型名称
            force_download: 是否强制重新下载
        
        Returns:
            (成功标志, 模型本地路径或错误信息)
        """
        try:
            model_path = self.get_model_path(model_name)
            
            # 如果模型已存在且不强制下载，直接返回
            if not force_download and self.is_model_downloaded(model_name):
                print(f"✅ 模型已存在，跳过下载: {model_path}")
                return True, str(model_path)
            
            print(f"🔄 开始下载模型: {model_name}")
            print(f"📥 下载目标目录: {model_path}")
            
            # 使用huggingface_hub下载模型
            from huggingface_hub import snapshot_download
            
            # 下载模型到指定目录
            downloaded_path = snapshot_download(
                repo_id=model_name,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
                resume_download=True,          # 支持断点续传
                max_workers=4,                 # 并行下载线程数
            )
            
            print(f"✅ 模型下载完成: {downloaded_path}")
            return True, str(model_path)
            
        except Exception as e:
            error_msg = f"❌ 模型下载失败: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def get_model_info(self, model_name: str) -> dict:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
        
        Returns:
            包含模型信息的字典
        """
        model_path = self.get_model_path(model_name)
        
        info = {
            "name": model_name,
            "local_path": str(model_path),
            "downloaded": self.is_model_downloaded(model_name),
            "exists": model_path.exists(),
        }
        
        # 如果模型已下载，获取更多信息
        if info["downloaded"]:
            try:
                # 计算模型大小
                total_size = 0
                for file_path in model_path.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                
                info["size_gb"] = total_size / (1024 ** 3)  # 转换为GB
                
                # 列出主要文件
                info["files"] = [f.name for f in model_path.iterdir() if f.is_file()]
                
            except Exception as e:
                info["error"] = str(e)
        
        return info
    
    def list_downloaded_models(self) -> list:
        """
        列出所有已下载的模型
        
        Returns:
            已下载模型的列表
        """
        if not self.models_dir.exists():
            return []
        
        downloaded = []
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                # 检查是否是有效的模型目录
                if (model_dir / "config.json").exists():
                    downloaded.append(model_dir.name)
        
        return downloaded
    
    def clear_cache(self, model_name: Optional[str] = None):
        """
        清除模型缓存
        
        Args:
            model_name: 要清除的模型名称，如果为None则清除所有
        """
        if model_name:
            model_path = self.get_model_path(model_name)
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
                print(f"🗑️ 已清除模型缓存: {model_path}")
        else:
            if self.models_dir.exists():
                import shutil
                shutil.rmtree(self.models_dir)
                self.models_dir.mkdir(parents=True, exist_ok=True)
                print(f"🗑️ 已清除所有模型缓存")


# 全局模型管理器实例
_global_model_manager = None


def get_model_manager(comfyui_path: str = "E:/Comfyui_test/ComfyUI") -> Sa2VAModelManager:
    """
    获取全局模型管理器实例
    
    Args:
        comfyui_path: ComfyUI根目录路径
    
    Returns:
        模型管理器实例
    """
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = Sa2VAModelManager(comfyui_path)
    return _global_model_manager
