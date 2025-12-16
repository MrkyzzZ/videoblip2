import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import cv2  # 确保导入cv2
from transformers import CLIPProcessor, CLIPModel

# 配置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 设置本地路径
MODEL_PATH = '/home/qinxilin/baseline/clip-vit-large-patch14-336'  # CLIP模型路径
VIDEO_FRAMES_PATH = '/home/qinxilin/baseline/MUCIS-AVQA-videos-Synthetic'  # 视频文件路径
OUTPUT_FEATURES_PATH = '/home/qinxilin/baseline/frame_ViT-L14@336px_patchlevel'  # 输出特征路径

# 加载CLIP模型（从本地文件加载）
def load_clip_model(model_path, device):
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    return model, processor

# 加载本地模型
model, processor = load_clip_model(MODEL_PATH, device)

# 获取patch数量：根据图像尺寸和patch大小来计算
def get_patch_count(image_size, patch_size):
    return (image_size // patch_size) ** 2

image_size = 336  # 图像大小，336x336
patch_size = 14   # patch的大小，14x14

# 计算patch数量
patch_nums = get_patch_count(image_size, patch_size)  # 576个patch
C = 1024  # 每个patch的特征维度（ViT的输出为1024）

# 提取单张图像的patch级别特征
def clip_feat_extract_patch_level(img):
    """
    获取每个patch的特征，不包含[CLS] token
    """
    image = processor(images=img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # 获取 ViT 输出的所有 patch 特征
        # 我们通过访问 ViT 中间层的输出
        outputs = model.vision_model(**image)  # VisionTransformer的输出
        hidden_states = outputs.last_hidden_state  # 输出是[batch_size, num_patches + 1, hidden_size]
        
        # shape: [1, num_patches+1, hidden_size]
        # 去掉[CLS] token，保留patch特征
        patch_features = hidden_states[:, 1:, :]  # 去掉[CLS] token，保留patch特征

    return patch_features

# 从MP4视频中提取每一秒的一帧
def extract_frames_from_video(video_path, frame_rate=1):
    """
    使用 OpenCV 从视频中提取每秒一帧
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    # 确保视频打开成功
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    frame_interval = int(fps // frame_rate)  # 每秒提取一帧

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 跳过非采样帧
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_interval != 0:
            continue

        # 将 BGR 图像转换为 RGB 格式（CLIP使用的是RGB图像）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames

# 提取视频中每一帧的patch级别特征
def ImageClIP_Patch_feat_extract(dir_fps_path, dst_clip_path):
    video_list = os.listdir(dir_fps_path)
    video_idx = 0
    total_nums = len(video_list)

    for video in video_list:
        video_idx += 1
        print("\n--> ", video_idx, video)
        save_file = os.path.join(dst_clip_path, video + '.npy')

        if os.path.exists(save_file):
            print(video + '.npy', "is already processed!")
            continue

        video_path = os.path.join(dir_fps_path, video)
        frames = extract_frames_from_video(video_path)

        if len(frames) == 0:
            print(f"Warning: No frames extracted from {video}. Skipping...")
            continue

        print(f"Video {video}: Extracted {len(frames)} frames")

        # 将每一帧图像转换为PIL图像格式并提取patch特征
        img_features = torch.zeros(len(frames), patch_nums, C)  # patch_nums = 576, C = 1024

        for idx, frame in enumerate(frames):
            # 将每一帧转换为PIL图像格式，然后提取patch特征
            pil_frame = Image.fromarray(frame)
            patch_idx_feat = clip_feat_extract_patch_level(pil_frame)
            img_features[idx] = patch_idx_feat

        # 转换为numpy格式并保存
        img_features = img_features.float().cpu().numpy()
        np.save(save_file, img_features)

        print("Process: ", video_idx, " / ", total_nums, " ----- video id: ", video_idx, " ----- save shape: ", img_features.shape)

# 主程序
if __name__ == "__main__":
    os.makedirs(OUTPUT_FEATURES_PATH, exist_ok=True)

    # 提取视频帧的patch特征
    ImageClIP_Patch_feat_extract(VIDEO_FRAMES_PATH, OUTPUT_FEATURES_PATH)






