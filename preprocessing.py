import cv2
import numpy as np

def preprocess_image(raw_bgr_image, target_w=640, target_h=640):
    """
    描述: 将BGR图像转换为RGB，
           调整大小并填充到目标尺寸，归一化为[0,1]，
           转换为NCHW格式。
    参数:
        raw_bgr_image: 原始BGR图像
        target_w: 目标宽度
        target_h: 目标高度
    返回:
        image: 处理后的图像
        image_raw: 原始图像
        h: 原始高度
        w: 原始宽度
        scale_factor: 缩放因子
        pad_info: 填充信息 (ty1, ty2, tx1, tx2)
    """
    image_raw = raw_bgr_image
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    
    # 计算宽度、高度和填充
    r_w = target_w / w
    r_h = target_h / h
    scale_factor = min(r_w, r_h)
    
    if r_h > r_w:  # 宽度限制因素
        tw = target_w
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((target_h - th) / 2)
        ty2 = target_h - th - ty1
    else:  # 高度限制因素
        tw = int(r_h * w)
        th = target_h
        tx1 = int((target_w - tw) / 2)
        tx2 = target_w - tw - tx1
        ty1 = ty2 = 0
        
    # 调整图像大小，保持长边比例
    image = cv2.resize(image, (tw, th))
    
    # 使用(128,128,128)填充短边
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
    )
    
    pad_info = (ty1, ty2, tx1, tx2)
    
    image = image.astype(np.float32)
    # 归一化为[0,1]
    image /= 255.0
    # HWC转CHW格式:
    image = np.transpose(image, [2, 0, 1])
    # CHW转NCHW格式
    image = np.expand_dims(image, axis=0)
    # 将图像转换为行主序，也称为"C顺序":
    image = np.ascontiguousarray(image)
    
    return image, image_raw, h, w, scale_factor, pad_info 