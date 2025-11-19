# Sa2VA 节点 for ComfyUI - 图像分割和理解
# 支持文本生成和分割掩码输出
# 基于 ByteDance/Sa2VA 模型，结合 SAM2 和 LLaVA

import torch
import numpy as np
import os
import gc
from contextlib import nullcontext
from PIL import Image
from typing import Tuple, List, Optional

# 导入模型管理器
import sys
from pathlib import Path

# 添加父目录到路径以导入model_manager
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from model_manager import get_model_manager


class Sa2VANode:
    """Sa2VA ComfyUI节点 - 图像分割和视觉理解"""
    
    def __init__(self):
        """初始化节点"""
        self.model = None
        self.processor = None
        self.current_model_name = None  # 跟踪当前加载的模型
        self.model_manager = get_model_manager()  # 获取模型管理器
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义节点的输入类型"""
        return {
            "required": {
                # 🖼️ 图像输入
                "🖼️图像": ("IMAGE",),
                
                # 🤖 模型选择
                "🤖模型选择": (
                    [
                        "Sa2VA-1B (1B参数)",
                        "Sa2VA-4B (4B参数)",
                        "Sa2VA-8B (8B参数)",
                        "Sa2VA-26B (26B参数)",
                        "Sa2VA-InternVL3-2B (2B参数)",
                        "Sa2VA-InternVL3-8B (8B参数)",
                        "Sa2VA-InternVL3-14B (14B参数)",
                        "Sa2VA-Qwen2.5-VL-3B (3B参数)",
                        "Sa2VA-Qwen2.5-VL-7B (7B参数)",
                        "Sa2VA-Qwen3-VL-4B (4B参数) ⭐推荐",
                    ],
                    {"default": "Sa2VA-Qwen3-VL-4B (4B参数) ⭐推荐"},
                ),
                
                # ⚙️ 量化级别
                "⚙️量化级别": (
                    [
                        "None (FP16/BF16)",
                        "4bit (NF4)",
                    ],
                    {"default": "None (FP16/BF16)"},
                ),
                
                # 💬 提示词
                "💬提示词": (
                    "STRING",
                    {
                        "default": "请描述这张图片，并为相应的部分提供分割掩码。",
                        "multiline": True,
                    },
                ),
                
                # 🎭 遮罩阈值
                "🎭遮罩阈值": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                
                # 💻 设备选择
                "💻设备选择": (
                    ["auto", "cuda", "cpu"],
                    {"default": "auto"},
                ),
                
                # 🎲 随机种子
                "🎲随机种子": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xffffffffffffffff},
                ),
                
                # 🎯 种子控制
                "🎯种子控制": (
                    ["固定", "随机", "递增"],
                    {"default": "固定"},
                ),
            },
            "optional": {
                # ⚡ Flash Attention
                "⚡启用FlashAttention": (
                    "BOOLEAN",
                    {"default": True},
                ),
                
                # 🔄 保持模型加载
                "🔄保持模型加载": (
                    "BOOLEAN",
                    {"default": False},
                ),
                
                # 🔃 强制重新下载
                "🔃强制重新下载": (
                    "BOOLEAN",
                    {"default": False},
                ),
                
                # 🎨 遮罩预处理
                "🎨启用遮罩预处理": (
                    "BOOLEAN",
                    {"default": False},
                ),
                
                # 📏 扩展（像素）
                "📏扩展": (
                    "INT",
                    {"default": 0, "min": -999, "max": 999, "step": 1},
                ),
                
                # 📐 扩展增量
                "📐扩展增量": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                
                # 🔲 倒角
                "🔲倒角": (
                    "BOOLEAN",
                    {"default": True},
                ),
                
                # 🔄 反转输入
                "🔄反转输入": (
                    "BOOLEAN",
                    {"default": False},
                ),
                
                # 🌫️ 模糊半径
                "🌫️模糊半径": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                
                # 💫 线性透明
                "💫线性透明": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                
                # 🎚️ 腐蚀系数
                "🎚️腐蚀系数": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                
                # 🔳 填补
                "🔳填补": (
                    "BOOLEAN",
                    {"default": False},
                ),
            },
        }
    
    RETURN_TYPES = ("STRING", "MASK", "IMAGE")
    RETURN_NAMES = ("📊结果分析", "分割遮罩", "遮罩图像")
    FUNCTION = "process"
    CATEGORY = "🤖大炮-Sa2VA"
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """
        检查依赖项是否满足
        
        Returns:
            (是否满足, 错误信息)
        """
        try:
            # 检查transformers版本
            from transformers import __version__ as transformers_version
            
            version_parts = transformers_version.split(".")
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            # Sa2VA需要transformers >= 4.57.0
            if major < 4 or (major == 4 and minor < 57):
                return (
                    False,
                    f"Sa2VA需要 transformers >= 4.57.0，当前版本: {transformers_version}\n"
                    f"请运行: pip install transformers>=4.57.0 --upgrade"
                )
            
            return True, transformers_version
            
        except Exception as e:
            return False, f"检查依赖时出错: {e}"
    
    def load_model(
        self,
        model_name: str,
        quantization_level: str = "None (FP16/BF16)",
        device_choice: str = "auto",
        use_flash_attn: bool = True,
        force_download: bool = False,
        keep_model_loaded: bool = False,
    ) -> bool:
        """
        加载Sa2VA模型
        
        Args:
            model_name: 模型名称（显示名称）
            quantization_level: 量化级别
            device_choice: 设备选择
            use_flash_attn: 是否使用Flash Attention
            force_download: 是否强制重新下载
            keep_model_loaded: 是否保持模型加载
        
        Returns:
            是否加载成功
        """
        # 如果模型已加载且是同一个模型，并且保持加载，直接返回
        if (
            keep_model_loaded
            and self.model is not None
            and self.processor is not None
            and self.current_model_name == model_name
        ):
            print(f"✅ 模型已加载（保持加载模式）: {model_name}")
            return True
        
        # 清理旧模型
        if self.model is not None:
            try:
                del self.model
                self.model = None
            except:
                pass
        
        if self.processor is not None:
            try:
                del self.processor
                self.processor = None
            except:
                pass
        
        self.current_model_name = None
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        
        # 转换显示名称为实际模型名称
        model_name_map = {
            "Sa2VA-1B (1B参数)": "ByteDance/Sa2VA-1B",
            "Sa2VA-4B (4B参数)": "ByteDance/Sa2VA-4B",
            "Sa2VA-8B (8B参数)": "ByteDance/Sa2VA-8B",
            "Sa2VA-26B (26B参数)": "ByteDance/Sa2VA-26B",
            "Sa2VA-InternVL3-2B (2B参数)": "ByteDance/Sa2VA-InternVL3-2B",
            "Sa2VA-InternVL3-8B (8B参数)": "ByteDance/Sa2VA-InternVL3-8B",
            "Sa2VA-InternVL3-14B (14B参数)": "ByteDance/Sa2VA-InternVL3-14B",
            "Sa2VA-Qwen2.5-VL-3B (3B参数)": "ByteDance/Sa2VA-Qwen2_5-VL-3B",
            "Sa2VA-Qwen2.5-VL-7B (7B参数)": "ByteDance/Sa2VA-Qwen2_5-VL-7B",
            "Sa2VA-Qwen3-VL-4B (4B参数) ⭐推荐": "ByteDance/Sa2VA-Qwen3-VL-4B",
        }
        
        actual_model_name = model_name_map.get(model_name, model_name)
        print(f"🔄 开始加载模型: {actual_model_name}")
        
        # 检查依赖
        deps_ok, deps_info = self.check_dependencies()
        if not deps_ok:
            print(f"❌ {deps_info}")
            return False
        
        print(f"✅ Transformers版本检查通过: {deps_info}")
        
        try:
            # 使用模型管理器下载模型
            success, model_path = self.model_manager.download_model(
                actual_model_name, 
                force_download=force_download
            )
            
            if not success:
                print(f"❌ 模型下载失败: {model_path}")
                return False
            
            print(f"📁 模型路径: {model_path}")
            
            # 导入transformers
            from transformers import AutoProcessor, AutoModel
            
            # 准备模型加载参数
            model_kwargs = {
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            
            # 量化配置
            use_quantization = quantization_level != "None (FP16/BF16)"
            if use_quantization:
                try:
                    from transformers import BitsAndBytesConfig
                    
                    if quantization_level == "4bit (NF4)":
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                        print("✅ 启用4位量化 (NF4)")
                    
                    model_kwargs["quantization_config"] = quantization_config
                except ImportError:
                    print("⚠️ bitsandbytes未安装，跳过量化")
                    print("   安装命令: pip install bitsandbytes")
                    use_quantization = False
            
            # Flash Attention配置
            if use_flash_attn:
                try:
                    import flash_attn
                    model_kwargs["use_flash_attn"] = True
                    print("✅ 启用Flash Attention")
                except ImportError:
                    print("⚠️ flash-attn未安装，跳过Flash Attention")
                    print("   安装命令: pip install flash-attn")
            
            # 确定目标设备
            if device_choice == "auto":
                target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif device_choice == "cuda":
                if torch.cuda.is_available():
                    target_device = torch.device("cuda")
                else:
                    print("⚠️ CUDA不可用，回退到CPU")
                    target_device = torch.device("cpu")
            else:
                target_device = torch.device("cpu")
            
            print(f"💻 目标设备: {target_device}")
            
            # 设置数据类型
            if not use_quantization:
                if target_device.type == "cuda":
                    # 优先使用bfloat16，如果不支持则使用float16
                    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                        model_kwargs["torch_dtype"] = torch.bfloat16
                        print("✅ 使用bfloat16精度")
                    else:
                        model_kwargs["torch_dtype"] = torch.float16
                        print("✅ 使用float16精度")
                else:
                    model_kwargs["torch_dtype"] = torch.float32
                    print("✅ 使用float32精度（CPU模式）")
            
            # 从本地路径加载模型
            print("🔄 正在加载模型权重...")
            self.model = AutoModel.from_pretrained(
                model_path,
                **model_kwargs
            ).eval()
            
            # 移动模型到设备
            if not use_quantization:  # 量化会自动处理设备
                self.model = self.model.to(target_device)
                print(f"✅ 模型已移动到: {target_device}")
            
            # 加载处理器
            print("🔄 正在加载处理器...")
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
            )
            
            self.current_model_name = model_name
            
            print(f"✅ 模型加载完成: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ 加载模型时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_image(
        self,
        image: Image.Image,
        prompt: str,
    ) -> Tuple[str, List]:
        """
        处理单张图像
        
        Args:
            image: PIL图像
            prompt: 提示词
        
        Returns:
            (文本输出, 分割掩码列表)
        """
        try:
            # 准备输入
            input_dict = {
                "image": image,
                "text": f"<image>{prompt}",
                "past_text": "",
                "mask_prompts": None,
                "processor": self.processor,
            }
            
            # 推理
            with torch.no_grad():
                return_dict = self.model.predict_forward(**input_dict)
            
            # 提取结果
            text_output = return_dict.get("prediction", "")
            masks = return_dict.get("prediction_masks", [])
            
            return text_output, masks
            
        except Exception as e:
            error_msg = f"处理图像时出错: {e}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg, []
    
    def _generate_analysis_report(
        self,
        model_name: str,
        quantization_level: str,
        device_choice: str,
        image_size: tuple,
        num_masks: int,
        mask_threshold: float,
        process_time: float,
        total_time: float,
        seed: int,
        seed_control: str,
        model_output: str
    ) -> str:
        """
        生成详细的结果分析报告
        
        Args:
            model_name: 模型名称
            quantization_level: 量化级别
            device_choice: 设备选择
            image_size: 图像尺寸 (width, height)
            num_masks: 检测到的掩码数量
            mask_threshold: 掩码阈值
            process_time: 模型处理时间
            total_time: 总处理时间
            seed: 随机种子
            seed_control: 种子控制模式
            model_output: 模型文本输出
        
        Returns:
            格式化的分析报告
        """
        w, h = image_size
        
        # 获取设备信息
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_info = f"CUDA ({device_name})"
        else:
            device_info = "CPU"
        
        # 构建报告
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                   Sa2VA 执行结果分析                          ║
╚══════════════════════════════════════════════════════════════╝

✅ 执行状态: 成功完成

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 执行配置
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤖 使用模型: {model_name}
⚙️ 量化级别: {quantization_level}
💻 运行设备: {device_info} (选择: {device_choice})
📐 图像尺寸: {w} × {h} 像素

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 分割结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎭 掩码阈值: {mask_threshold}
   说明: 掩码阈值用于将模型输出的连续值掩码转换为二值掩码
         - 阈值范围: 0.0 到 1.0
         - 像素值 > {mask_threshold} → 前景 (白色, 值=1)
         - 像素值 ≤ {mask_threshold} → 背景 (黑色, 值=0)
         - 阈值越高，分割越严格，保留的区域越少
         - 阈值越低，分割越宽松，保留的区域越多
         - 推荐值: 0.5 (默认)

✅ 检测到掩码数量: {num_masks} 个
   - 每个掩码对应模型识别的一个物体或区域
   - 掩码已根据阈值 {mask_threshold} 进行二值化处理

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏱️ 性能统计
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔄 模型推理时间: {process_time:.2f} 秒
📦 总处理时间: {total_time:.2f} 秒
⚡ 处理速度: {1/total_time:.2f} 张/秒

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎲 随机种子信息
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 种子控制: {seed_control}
🎲 使用种子: {seed}
   说明: 使用相同的种子和参数可以重现相同的结果

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💬 模型输出摘要
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{model_output[:200]}{"..." if len(model_output) > 200 else ""}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ 输出说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 结果分析: 本报告（当前输出）
🎭 分割掩码: 二值化掩码张量，可用于后续处理
🖼️ 掩码图像: 可视化的掩码图像，可直接预览

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return report.strip()
    
    def preprocess_mask(
        self,
        mask: torch.Tensor,
        expand: int = 0,
        incremental_expand: float = 0.0,
        tapered_corners: bool = True,
        invert_input: bool = False,
        blur_radius: float = 0.0,
        lerp_alpha: float = 1.0,
        decay_factor: float = 1.0,
        fill_holes: bool = False,
    ) -> torch.Tensor:
        """
        遮罩预处理函数（参考KJNodes的遮罩模糊生长）
        
        Args:
            mask: 输入遮罩 (H, W)
            expand: 扩展像素数
            incremental_expand: 扩展增量
            tapered_corners: 是否倒角
            invert_input: 是否反转输入
            blur_radius: 模糊半径
            lerp_alpha: 线性透明度
            decay_factor: 腐蚀系数
            fill_holes: 是否填补孔洞
        
        Returns:
            处理后的遮罩
        """
        try:
            import cv2
            from scipy.ndimage import binary_fill_holes, distance_transform_edt
            
            # 转换为numpy
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = mask.copy()
            
            # 确保是2D
            if len(mask_np.shape) == 3:
                mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np[:, :, 0]
            
            # 反转输入
            if invert_input:
                mask_np = 1.0 - mask_np
            
            # 填补孔洞
            if fill_holes:
                mask_bool = mask_np > 0.5
                mask_filled = binary_fill_holes(mask_bool)
                mask_np = mask_filled.astype(np.float32)
            
            # 扩展/腐蚀
            total_expand = expand + incremental_expand
            if abs(total_expand) > 0:
                kernel_size = int(abs(total_expand) * 2) + 1
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE if tapered_corners else cv2.MORPH_RECT,
                    (kernel_size, kernel_size)
                )
                
                if total_expand > 0:
                    # 膨胀
                    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                else:
                    # 腐蚀
                    mask_np = cv2.erode(mask_np, kernel, iterations=1)
            
            # 应用腐蚀系数
            if decay_factor < 1.0:
                # 计算距离变换
                binary_mask = (mask_np > 0.5).astype(np.uint8)
                dist_transform = distance_transform_edt(binary_mask)
                
                # 归一化距离
                if dist_transform.max() > 0:
                    dist_norm = dist_transform / dist_transform.max()
                    # 应用衰减
                    mask_np = mask_np * (dist_norm ** (1.0 - decay_factor))
            
            # 模糊
            if blur_radius > 0:
                kernel_size = int(blur_radius * 2) + 1
                if kernel_size % 2 == 0:
                    kernel_size += 1
                mask_np = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), blur_radius / 2)
            
            # 线性插值（透明度）
            if lerp_alpha < 1.0:
                original = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                if len(original.shape) == 3:
                    original = original[0] if original.shape[0] == 1 else original[:, :, 0]
                mask_np = original * (1.0 - lerp_alpha) + mask_np * lerp_alpha
            
            # 确保在0-1范围
            mask_np = np.clip(mask_np, 0.0, 1.0)
            
            # 转回torch
            return torch.from_numpy(mask_np).float()
            
        except Exception as e:
            print(f"⚠️ 遮罩预处理失败: {e}")
            return mask
    
    def convert_masks_to_comfyui(
        self,
        masks: List,
        height: int,
        width: int,
        threshold: float = 0.5,
        enable_preprocess: bool = False,
        expand: int = 0,
        incremental_expand: float = 0.0,
        tapered_corners: bool = True,
        invert_input: bool = False,
        blur_radius: float = 0.0,
        lerp_alpha: float = 1.0,
        decay_factor: float = 1.0,
        fill_holes: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将Sa2VA的掩码转换为ComfyUI格式
        
        Args:
            masks: Sa2VA输出的掩码列表
            height: 图像高度
            width: 图像宽度
            threshold: 二值化阈值
        
        Returns:
            (掩码张量, 掩码图像张量)
        """
        try:
            # 如果没有掩码，返回空掩码
            if masks is None or len(masks) == 0:
                print("⚠️ 没有检测到掩码，返回空掩码")
                empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
                empty_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
                return empty_mask, empty_image
            
            comfyui_masks = []
            mask_images = []
            
            for i, mask in enumerate(masks):
                if mask is None:
                    continue
                
                try:
                    # 转换为numpy数组
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask.detach().cpu().numpy()
                    elif isinstance(mask, np.ndarray):
                        mask_np = mask.copy()
                    else:
                        continue
                    
                    # 处理不同的维度
                    if len(mask_np.shape) == 4:  # (batch, channel, height, width)
                        mask_np = mask_np[0, 0]
                    elif len(mask_np.shape) == 3:
                        if mask_np.shape[0] == 1:  # (1, height, width)
                            mask_np = mask_np[0]
                        elif mask_np.shape[2] == 1:  # (height, width, 1)
                            mask_np = mask_np[:, :, 0]
                        else:
                            mask_np = mask_np[0]
                    
                    # 确保是2D掩码
                    if len(mask_np.shape) != 2:
                        continue
                    
                    # 转换为float32
                    if mask_np.dtype == bool:
                        mask_np = mask_np.astype(np.float32)
                    elif not np.issubdtype(mask_np.dtype, np.floating):
                        mask_np = mask_np.astype(np.float32)
                    
                    # 处理NaN和无穷值
                    if np.any(np.isnan(mask_np)) or np.any(np.isinf(mask_np)):
                        mask_np = np.nan_to_num(mask_np, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    # 归一化到0-1
                    mask_min, mask_max = mask_np.min(), mask_np.max()
                    if mask_max > mask_min:
                        mask_np = (mask_np - mask_min) / (mask_max - mask_min)
                    else:
                        mask_np = np.ones_like(mask_np) if mask_min > 0 else np.zeros_like(mask_np)
                    
                    # 应用阈值
                    mask_np = (mask_np > threshold).astype(np.float32)
                    
                    # 转换为torch张量
                    mask_tensor = torch.from_numpy(mask_np).float()
                    
                    # 应用预处理（如果启用）
                    if enable_preprocess:
                        mask_tensor = self.preprocess_mask(
                            mask_tensor,
                            expand=expand,
                            incremental_expand=incremental_expand,
                            tapered_corners=tapered_corners,
                            invert_input=invert_input,
                            blur_radius=blur_radius,
                            lerp_alpha=lerp_alpha,
                            decay_factor=decay_factor,
                            fill_holes=fill_holes,
                        )
                    
                    comfyui_masks.append(mask_tensor)
                    
                    # 创建RGB掩码图像
                    rgb_np = np.stack([mask_np, mask_np, mask_np], axis=-1)
                    rgb_np = np.clip(rgb_np, 0.0, 1.0).astype(np.float32)
                    mask_images.append(torch.from_numpy(rgb_np))
                    
                except Exception as e:
                    print(f"⚠️ 处理第{i}个掩码时出错: {e}")
                    continue
            
            # 如果没有成功处理的掩码
            if not comfyui_masks:
                empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
                empty_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
                return empty_mask, empty_image
            
            # 堆叠掩码
            final_masks = torch.stack(comfyui_masks, dim=0)  # (N, H, W)
            final_images = torch.stack(mask_images, dim=0)   # (N, H, W, 3)
            
            return final_masks, final_images
            
        except Exception as e:
            print(f"❌ 转换掩码时出错: {e}")
            import traceback
            traceback.print_exc()
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            empty_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return empty_mask, empty_image
    
    def process(
        self,
        **kwargs
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        主处理函数
        
        Args:
            **kwargs: 所有输入参数
        
        Returns:
            (文本输出, 掩码张量, 掩码图像张量)
        """
        try:
            # 提取参数（使用中文键名）
            image = kwargs.get("🖼️图像")
            model_name = kwargs.get("🤖模型选择")
            quantization_level = kwargs.get("⚙️量化级别", "None (FP16/BF16)")
            prompt = kwargs.get("💬提示词")
            mask_threshold = kwargs.get("🎭遮罩阈值", 0.5)
            device_choice = kwargs.get("💻设备选择", "auto")
            seed = kwargs.get("🎲随机种子", 0)
            seed_control = kwargs.get("🎯种子控制", "固定")
            use_flash_attn = kwargs.get("⚡启用FlashAttention", True)
            keep_model_loaded = kwargs.get("🔄保持模型加载", False)
            force_download = kwargs.get("🔃强制重新下载", False)
            
            # 遮罩预处理参数
            enable_preprocess = kwargs.get("🎨启用遮罩预处理", False)
            expand = kwargs.get("📏扩展", 0)
            incremental_expand = kwargs.get("📐扩展增量", 0.0)
            tapered_corners = kwargs.get("🔲倒角", True)
            invert_input = kwargs.get("🔄反转输入", False)
            blur_radius = kwargs.get("🌫️模糊半径", 0.0)
            lerp_alpha = kwargs.get("💫线性透明", 1.0)
            decay_factor = kwargs.get("🎚️腐蚀系数", 1.0)
            fill_holes = kwargs.get("🔳填补", False)
            
            # 处理随机种子
            if seed_control == "随机":
                import random
                seed = random.randint(0, 0xffffffffffffffff)
                print(f"🎲 使用随机种子: {seed}")
            elif seed_control == "递增":
                if not hasattr(self, '_last_seed'):
                    self._last_seed = seed
                else:
                    self._last_seed += 1
                seed = self._last_seed
                print(f"🎲 使用递增种子: {seed}")
            else:
                print(f"🎲 使用固定种子: {seed}")
            
            # 设置随机种子
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2**32))
            
            # 加载模型
            if not self.load_model(
                model_name, 
                quantization_level,
                device_choice,
                use_flash_attn, 
                force_download,
                keep_model_loaded
            ):
                error_msg = f"模型加载失败: {model_name}"
                print(f"❌ {error_msg}")
                # 返回错误信息和空掩码
                h, w = 512, 512
                if hasattr(image, "shape") and len(image.shape) >= 2:
                    if len(image.shape) == 4:
                        h, w = image.shape[1], image.shape[2]
                    elif len(image.shape) == 3:
                        h, w = image.shape[0], image.shape[1]
                empty_mask = torch.zeros((1, h, w), dtype=torch.float32)
                empty_image = torch.zeros((1, h, w, 3), dtype=torch.float32)
                return error_msg, empty_mask, empty_image
            
            # 验证输入
            if image is None:
                error_msg = "未提供图像"
                print(f"⚠️ {error_msg}")
                empty_mask = torch.zeros((1, 512, 512), dtype=torch.float32)
                empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return error_msg, empty_mask, empty_image
            
            print(f"🔄 开始处理图像...")
            
            # 转换ComfyUI图像为PIL图像
            if hasattr(image, "shape") and len(image.shape) == 4:
                # ComfyUI格式: (batch, height, width, channels)
                img_tensor = image[0]
            elif hasattr(image, "shape") and len(image.shape) == 3:
                # 单张图像: (height, width, channels)
                img_tensor = image
            else:
                error_msg = f"不支持的图像格式: {type(image)}"
                print(f"❌ {error_msg}")
                empty_mask = torch.zeros((1, 512, 512), dtype=torch.float32)
                empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return error_msg, empty_mask, empty_image
            
            # 转换为numpy
            if isinstance(img_tensor, torch.Tensor):
                img_tensor = img_tensor.detach().cpu()
                image_np = img_tensor.numpy()
            else:
                error_msg = f"不支持的张量类型: {type(image)}"
                print(f"❌ {error_msg}")
                empty_mask = torch.zeros((1, 512, 512), dtype=torch.float32)
                empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return error_msg, empty_mask, empty_image
            
            # 转换为uint8
            if image_np.dtype != "uint8":
                image_np = (image_np * 255).astype("uint8")
            
            # 转换为PIL图像
            pil_image = Image.fromarray(image_np)
            h, w = image_np.shape[0], image_np.shape[1]
            
            print(f"📐 图像尺寸: {w}x{h}")
            
            # 记录开始时间
            import time
            start_time = time.time()
            
            # 处理图像
            with torch.inference_mode():
                text_output, masks = self.process_image(pil_image, prompt)
            
            # 记录处理时间
            process_time = time.time() - start_time
            
            print(f"✅ 模型输出: {text_output[:100]}...")  # 只打印前100个字符
            print(f"✅ 检测到 {len(masks)} 个掩码")
            
            # 转换掩码
            comfyui_masks, mask_images = self.convert_masks_to_comfyui(
                masks, h, w, mask_threshold,
                enable_preprocess=enable_preprocess,
                expand=expand,
                incremental_expand=incremental_expand,
                tapered_corners=tapered_corners,
                invert_input=invert_input,
                blur_radius=blur_radius,
                lerp_alpha=lerp_alpha,
                decay_factor=decay_factor,
                fill_holes=fill_holes,
            )
            
            # 计算总时间
            total_time = time.time() - start_time
            
            print(f"✅ 处理完成")
            print(f"   掩码形状: {comfyui_masks.shape}")
            print(f"   掩码图像形状: {mask_images.shape}")
            
            # 生成详细的结果分析报告
            analysis_report = self._generate_analysis_report(
                model_name=model_name,
                quantization_level=quantization_level,
                device_choice=device_choice,
                image_size=(w, h),
                num_masks=len(masks),
                mask_threshold=mask_threshold,
                process_time=process_time,
                total_time=total_time,
                seed=seed,
                seed_control=seed_control,
                model_output=text_output
            )
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return analysis_report, comfyui_masks, mask_images
            
        except Exception as e:
            error_msg = f"处理失败: {e}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            
            # 返回错误和空掩码
            h, w = 512, 512
            try:
                if hasattr(image, "shape") and len(image.shape) >= 2:
                    if len(image.shape) == 4:
                        h, w = image.shape[1], image.shape[2]
                    elif len(image.shape) == 3:
                        h, w = image.shape[0], image.shape[1]
            except:
                pass
            
            empty_mask = torch.zeros((1, h, w), dtype=torch.float32)
            empty_image = torch.zeros((1, h, w, 3), dtype=torch.float32)
            return f"错误: {error_msg}", empty_mask, empty_image
