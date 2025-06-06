import cv2
import numpy as np
import random
from utils import colors, class_names

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    将一个边界框绘制到图像上，来源于YoLov8项目
    
    参数:
        x:              边界框坐标，格式为[x1,y1,x2,y2]
        img:            opencv图像对象
        color:          绘制矩形的颜色，例如(0,255,0)
        label:          边界框标签
        line_thickness: 线条粗细
    返回:
        无返回值
    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # 线条/字体粗细
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # 字体粗细
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # 填充
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

def draw_box(image: np.ndarray, box: np.ndarray, color: tuple = (0, 0, 255),
             thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制边界框
    
    参数:
        image: 图像
        box: 边界框坐标[x1, y1, x2, y2]
        color: 颜色
        thickness: 线条粗细
    返回:
        绘制了边界框的图像
    """
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def draw_text(image: np.ndarray, text: str, box: np.ndarray, color: tuple = (0, 0, 255),
              font_size: float = 0.001, text_thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制文本
    
    参数:
        image: 图像
        text: 文本内容
        box: 边界框坐标[x1, y1, x2, y2]
        color: 颜色
        font_size: 字体大小
        text_thickness: 文本粗细
    返回:
        绘制了文本的图像
    """
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1),
                  (x1 + tw, y1 - th), color, -1)

    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    """
    在图像上绘制检测结果
    
    参数:
        image: 原始图像
        boxes: 边界框坐标数组
        scores: 置信度分数数组
        class_ids: 类别ID数组
        mask_alpha: 遮罩透明度
    返回:
        绘制了检测结果的图像
    """
    # 确保图像是可写的副本
    det_img = image.copy()

    # 如果没有检测到对象，直接返回原图
    if len(boxes) == 0:
        return det_img

    img_height, img_width = image.shape[:2]
    font_size = max(0.5, min([img_height, img_width]) * 0.0006)  # 确保字体大小最小为0.5
    text_thickness = max(1, int(min([img_height, img_width]) * 0.001))  # 确保文本线条至少为1

    # 绘制检测的边界框和标签
    for i, (class_id, box, score) in enumerate(zip(class_ids, boxes, scores)):
        # 确保class_id在范围内
        class_id = int(class_id) % len(colors)
        # 将颜色从float转换为int元组
        color = tuple(map(int, colors[class_id]))
        # 确保边界框坐标不越界
        box = np.clip(box, 0, max(img_width, img_height))
        
        # 绘制边界框
        draw_box(det_img, box, color, 2)
        
        # 获取类别名称和准备标题
        label = class_names[class_id] if class_id < len(class_names) else "unknown"
        caption = f'{label} {int(score * 100)}%'
        
        # 绘制标题
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img 