import torch
from torchvision import transforms
import cv2

def video_transforms(video_path ,img_size,n_frame=50):
    # 定义转换
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = transforms.Resize(img_size)  # Resize to (470, 800)
    # 使用 OpenCV 打开视频
    cap = cv2.VideoCapture(video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error opening video file")
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 列表来存储每一帧的处理结果
    frames = []
    if total_frames < n_frame:
        while True:
            ret, frame = cap.read()  # 读取视频的每一帧
            if not ret:
                break  # 如果没有帧可读，则结束循环
            # OpenCV 读取的图像是 BGR 格式，转换为 RGB 格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 将图像转换为 PIL Image
            frame_pil = transforms.ToPILImage()(frame)
            frame_resized = resize(frame_pil)
            # 转换为 Tensor
            frame_tensor = transforms.ToTensor()(frame_resized)
            # 归一化处理
            frame_normalized = normalize(frame_tensor)
            # 添加到帧列表
            frames.append(frame_normalized)

        # 如果帧数少于 n_frame，重复帧
        frames *= (n_frame // len(frames)) + 1  # 计算需要重复多少次
        # print(len(frames))
        frames = frames[:n_frame]  # 截取前 n_frame 帧
    else:
        frame_indices = torch.linspace(0, total_frames - 1, n_frame).long()  # 计算均匀采样的索引
        for index in frame_indices:
            index =int(index)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)  # 设置当前帧的位置
            ret, frame = cap.read()  # 读取指定帧

            if not ret:
                print(f"Error reading frame at index {index}")
                continue  # 如果读取失败，跳过

            # OpenCV 读取的图像是 BGR 格式，转换为 RGB 格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 将图像转换为 PIL Image
            frame_pil = transforms.ToPILImage()(frame)
            # Resize 图像到 (470, 800)
            frame_resized = resize(frame_pil)
            # 转换为 Tensor
            frame_tensor = transforms.ToTensor()(frame_resized)
            # 归一化处理
            frame_normalized = normalize(frame_tensor)
            # 添加到帧列表
            frames.append(frame_normalized)

            # 释放视频捕获对象
    cap.release()
    video_tensor = torch.stack(frames)
    # # 变换形状为 (3, n_frame, 470, 800)
    # video_tensor = video_tensor.permute(1, 0, 2, 3)
    return video_tensor