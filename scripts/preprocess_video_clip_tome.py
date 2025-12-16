import os
import torch
import torch.nn as nn
import numpy as np
import glob
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers.models.clip.modeling_clip import CLIPEncoderLayer

# --- (新) 添加路径并导入本地 Tome ---
import sys
import importlib.util
from pathlib import Path

# 直接按文件路径加载 tome/merge.py，绕过 __init__ 中的 src.* 依赖
TOME_MERGE_PATH = Path(__file__).resolve().parents[1] / "tome" / "merge.py"
if not TOME_MERGE_PATH.exists():
    print("=" * 50)
    print(f"错误: 找不到 ToMe 源文件: {TOME_MERGE_PATH}")
    print("请确认仓库中存在 tome/merge.py")
    print("=" * 50)
    sys.exit(1)

spec = importlib.util.spec_from_file_location("tome_merge", TOME_MERGE_PATH)
tome_merge = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(tome_merge)  # type: ignore
except Exception as e:
    print("=" * 50)
    print(f"错误: 加载 ToMe 模块失败: {e}")
    print(f"尝试加载的路径: {TOME_MERGE_PATH}")
    print("=" * 50)
    sys.exit(1)
# ------------------------------------

# 配置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = '/root/autodl-tmp/videoblip2/pretrained/clip-vit-large-patch14-336'
VIDEO_FRAMES_PATH = '/root/autodl-tmp/videoblip2/data/MSRVTT/video'
OUTPUT_FEATURES_PATH = '/root/autodl-tmp/videoblip2/data/MSRVTT_patchlevel_Vit'

R_VALUE_PER_LAYER = 18
VIT_L_LAYERS = 24

image_size = 336
patch_size = 14
original_patch_nums = (image_size // patch_size) ** 2  # 576
C = 1024  # ViT-L 隐藏层维度

new_patch_nums = original_patch_nums - (VIT_L_LAYERS * R_VALUE_PER_LAYER)  # 144

# ==========================================================================
# --- (关键修正) ---
#  1. 修正 ToMeBlock 的 forward 逻辑
# ==========================================================================
class ToMeBlock(nn.Module):
    """
    一个包装模块，它将 ToMe 逻辑注入到 HF CLIPEncoderLayer 中。
    它执行: Attention -> ToMe Merge -> MLP
    """
    def __init__(self, original_layer: CLIPEncoderLayer):
        super().__init__()
        
        self.self_attn = original_layer.self_attn
        self.layer_norm1 = original_layer.layer_norm1
        self.mlp = original_layer.mlp
        self.layer_norm2 = original_layer.layer_norm2
        
        self.head_dim = C // 16 
        self.r = 0 

    def forward(
        self, 
        hidden_states, 
        attention_mask, 
        causal_attention_mask, 
        output_attentions: bool = False
    ):
        # 1. --- 原始 Attention 部分 ---
        residual = hidden_states
        hidden_states_norm = self.layer_norm1(hidden_states)
        
        attn_outputs_tuple = self.self_attn(
            hidden_states=hidden_states_norm,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions 
        )
        
        attn_output = attn_outputs_tuple[0]
        hidden_states = attn_output + residual
        
        # 2. --- (新) ToMe 融合步骤 ---
        if self.r > 0 and hidden_states.shape[1] > (new_patch_nums + 1):
            
            with torch.no_grad():
                k = self.self_attn.k_proj(hidden_states_norm)
                k = k.view(k.shape[0], k.shape[1], 16, self.head_dim).permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3).flatten(2)
            
            try:
                # --- (修改点 1: 解包元组) ---
                # bipartite_soft_matching 返回 (merge_fn, unmerge_fn)
                # 我们只取第一个，即 merge_fn
                merge_fn, _ = tome_merge.bipartite_soft_matching(
                    k, 
                    self.r,
                    class_token=True  # <-- (修改点 2: 必须!! 保护 CLS token)
                )
            except Exception as e:
                print(f"错误: 调用 tome.merge.bipartite_soft_matching 失败: {e}")
                sys.exit(1)
            
            try:
                # --- (修改点 3: 调用解包后的 merge_fn) ---
                # merge_fn 是一个需要 'x' 和 'mode' 的函数
                hidden_states = merge_fn(hidden_states, mode="mean")
            except Exception as e:
                print(f"错误: 调用返回的 'merge_fn' 失败: {e}")
                sys.exit(1)


        # 3. --- 原始 MLP 部分 ---
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (attn_outputs_tuple[1],)
        
        return outputs
# ==========================================================================


# 2. 定义手动注入函数
def apply_tome_to_hf_clip(clip_vision_model, r_value):
    """
    手动遍历 HF CLIP Vision Model 并用我们的 ToMeBlock 替换原始层
    """
    print("Applying ToMe patch MANUALLY to HuggingFace CLIP Vision Model...")
    
    original_layers = list(clip_vision_model.encoder.layers)
    clip_vision_model.encoder.layers = nn.ModuleList()
    
    for i, original_layer in enumerate(original_layers):
        new_block = ToMeBlock(original_layer)
        new_block.r = r_value
        clip_vision_model.encoder.layers.append(new_block)

    print(f"Successfully patched {len(clip_vision_model.encoder.layers)} layers.")
    return clip_vision_model


# --- 加载本地的 CLIP 模型并应用 ToMe ---
def load_clip_model_and_apply_tome(model_path, device):
    print(f"Loading local CLIP model from: {model_path}")
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    
    model.vision_model = apply_tome_to_hf_clip(model.vision_model, R_VALUE_PER_LAYER)
    
    model.eval()
    print(f"Model patched with ToMe. Each layer will merge r={R_VALUE_PER_LAYER} tokens.")
    
    # 检查输出形状
    check_input = processor(
        images=Image.new("RGB", (image_size, image_size)), 
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        check_output = model.vision_model(**check_input).last_hidden_state
    print(f"Output shape check (Batch, Tokens_with_CLS, Dim): {check_output.shape}")
    
    expected_tokens = original_patch_nums + 1 - (VIT_L_LAYERS * R_VALUE_PER_LAYER) # 576+1 - (24*18) = 145
    if check_output.shape[1] != expected_tokens:
         print(f"警告: 输出 Token 数量与预期不符! 预期: {expected_tokens}, 得到: {check_output.shape[1]}")
         # 这是一个渐进式合并，最终数量可能不完全精确，我们检查它是否在合理范围内
         if abs(check_output.shape[1] - expected_tokens) < 5:
             print("...但差异很小，在可接受范围内。")
             # (重要) 更新 new_patch_nums 为实际输出，以避免保存时出错
             global new_patch_nums
             new_patch_nums = check_output.shape[1] - 1 # 减去 CLS token
             print(f"!!! 已将 'new_patch_nums' 更新为实际值: {new_patch_nums} !!!")
         else:
             print("!!! 错误: Token 数量差异过大，请检查 'r' 值或模型结构。")
             sys.exit(1)
    else:
         print(f"Token 数量验证成功: {check_output.shape[1]}")

    return model, processor

# 加载并修补模型
model, processor = load_clip_model_and_apply_tome(MODEL_PATH, device)


# --- 提取单张图像的 patch 级别特征 (使用 HF CLIP) ---
def clip_feat_extract_patch_level(pil_img):
    image = processor(images=pil_img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.vision_model(**image)
        hidden_states = outputs.last_hidden_state  
        patch_features = hidden_states[:, 1:, :] 
    return patch_features

# --- 从MP4视频中提取每一秒的一帧 (保持不变) ---
def extract_frames_from_video(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Error: Video file {video_path} has FPS=0.")
        return []
        
    frame_interval = int(fps // frame_rate)
    
    if frame_interval == 0:
        frame_interval = 1

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_count += 1

    cap.release()
    return frames

# --- 提取视频中每一帧的patch级别特征 (使用新模型) ---
def ImageClIP_Patch_feat_extract(dir_fps_path, dst_clip_path):
    video_list = os.listdir(dir_fps_path)
    video_idx = 0
    total_nums = len(video_list)

    for video in video_list:
        video_idx += 1
        print("\n--> ", video_idx, video)
        
        video_name_without_ext = os.path.splitext(video)[0]
        save_file = os.path.join(dst_clip_path, video_name_without_ext + '.npy')

        if os.path.exists(save_file):
            print(f"{save_file} is already processed!")
            continue

        video_path = os.path.join(dir_fps_path, video)
        frames = extract_frames_from_video(video_path)

        if len(frames) == 0:
            print(f"Warning: No frames extracted from {video}. Skipping...")
            continue

        print(f"Video {video}: Extracted {len(frames)} frames")

        img_features = torch.zeros(len(frames), new_patch_nums, C) 

        for idx, frame in enumerate(frames):
            pil_frame = Image.fromarray(frame)
            patch_idx_feat = clip_feat_extract_patch_level(pil_frame)
            
            # 验证形状是否正确
            if patch_idx_feat.shape[1] != new_patch_nums:
                print(f"Error: Mismatched patch count! Expected {new_patch_nums}, got {patch_idx_feat.shape[1]}")
                # 填充或截断 (以防万一)
                if patch_idx_feat.shape[1] > new_patch_nums:
                    patch_idx_feat = patch_idx_feat[:, :new_patch_nums, :]
                else:
                    padded_feat = torch.zeros(1, new_patch_nums, C).to(device)
                    padded_feat[:, :patch_idx_feat.shape[1], :] = patch_idx_feat
                    patch_idx_feat = padded_feat
            
            img_features[idx] = patch_idx_feat

        img_features = img_features.float().cpu().numpy()
        np.save(save_file, img_features)

        print(f"Process: {video_idx}/{total_nums} --- video id: {video_name_without_ext} --- save shape: {img_features.shape}")

# 主程序
if __name__ == "__main__":
    os.makedirs(OUTPUT_FEATURES_PATH, exist_ok=True)

    print(f"Starting ToMe feature extraction (Targeting 144 tokens).")
    print(f"Source Videos: {VIDEO_FRAMES_PATH}")
    print(f"Destination Features: {OUTPUT_FEATURES_PATH}")
    print(f"Original Patch Count: {original_patch_nums}")
    print(f"ToMe Merging Rate (r): {R_VALUE_PER_LAYER}")
    print(f"Final Merged Patch Count: {new_patch_nums}")
    print(f"Feature Dimension: {C}")

    ImageClIP_Patch_feat_extract(VIDEO_FRAMES_PATH, OUTPUT_FEATURES_PATH)