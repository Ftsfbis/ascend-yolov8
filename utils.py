import os
import glob
import numpy as np
import cv2
import random

# 类别名称列表
class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

# 为每个类创建颜色列表，每种颜色是一个由3个整数值组成的元组
# 使用固定种子以确保颜色一致性
random.seed(42)
colors = []
for i in range(len(class_names)):
    # 确保颜色足够鲜明，避免使用太暗或太亮的颜色
    r = random.randint(50, 200)
    g = random.randint(50, 200)
    b = random.randint(50, 200)
    colors.append((b, g, r))  # OpenCV使用BGR格式

def get_model_name(model_path):
    """
    从模型路径中提取模型名称
    
    参数:
        model_path: 模型路径
    返回:
        model_name: 模型名称（不含扩展名）
    """
    # 获取文件名（不含路径）
    model_filename = os.path.basename(model_path)
    # 获取文件名（不含扩展名）
    model_name = os.path.splitext(model_filename)[0]
    return model_name

def create_output_dir(model_path, infer_mode, output_root="runs/"):
    """
    创建输出目录
    
    参数:
        model_path: 模型路径
        infer_mode: 推理模式
        output_root: 输出根目录
    返回:
        output_dir: 输出目录路径
    """
    # 确保输出根目录存在
    os.makedirs(output_root, exist_ok=True)
    
    # 获取模型名称
    model_name = get_model_name(model_path)
    
    # 构建目录前缀
    dir_prefix = f"{model_name}+{infer_mode}+推理结果"
    
    # 查找现有的同类型目录
    existing_dirs = glob.glob(os.path.join(output_root, f"{dir_prefix}*"))
    
    # 生成新的目录名序号
    if not existing_dirs:
        new_index = 1
    else:
        # 提取现有目录的序号
        indices = []
        for dir_path in existing_dirs:
            dir_name = os.path.basename(dir_path)
            if dir_name == dir_prefix:  # 没有序号
                indices.append(0)
            else:
                try:
                    # 尝试提取序号部分
                    suffix = dir_name[len(dir_prefix):]
                    if suffix.isdigit():
                        indices.append(int(suffix))
                except:
                    pass
        
        if not indices:
            new_index = 1
        else:
            new_index = max(indices) + 1
    
    # 构建新目录名
    if new_index == 1 and not existing_dirs:
        output_dir = os.path.join(output_root, dir_prefix)
    else:
        output_dir = os.path.join(output_root, f"{dir_prefix}{new_index}")
    
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"创建输出目录: {output_dir}")
    
    return output_dir

def get_img_path_batches(batch_size, img_dir):
    """
    将图像路径按批次分组
    
    参数:
        batch_size: 批次大小
        img_dir: 图像目录
    返回:
        ret: 分批后的图像路径列表
    """
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret

def save_detection_results(output_dir, boxes, scores, class_ids, frame_count=None, image_name="detection_results.txt"):
    """
    保存检测结果到文本文件
    
    参数:
        output_dir: 输出目录
        boxes: 检测框
        scores: 置信度分数
        class_ids: 类别ID
        frame_count: 帧计数（用于视频）
        image_name: 保存的文件名
    返回:
        result_file: 结果文件路径
    """
    result_file = os.path.join(output_dir, image_name)
    with open(result_file, "w" if frame_count is None else "a") as f:
        if frame_count is None:
            f.write("class_id,类别名称,置信度,x1,y1,x2,y2\n")
        
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[class_id] if class_id < len(class_names) else "unknown"
            
            if frame_count is not None:
                f.write(f"{frame_count},{class_id},{class_name},{score:.4f},{x1},{y1},{x2},{y2}\n")
            else:
                f.write(f"{class_id},{class_name},{score:.4f},{x1},{y1},{x2},{y2}\n")
    
    return result_file 