import numpy as np

def xywh2xyxy(x):
    """
    将边界框从中心点坐标加宽高格式转换为左上右下角点坐标格式
    
    参数:
        x: 边界框数组，格式为[x_center, y_center, width, height]
    返回:
        y: 边界框数组，格式为[x1, y1, x2, y2]
    """
    # 复制输入数组
    y = np.copy(x)
    # 计算左上角x坐标 = 中心x - 宽度/2
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    # 计算左上角y坐标 = 中心y - 高度/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    # 计算右下角x坐标 = 中心x + 宽度/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    # 计算右下角y坐标 = 中心y + 高度/2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def rescale_boxes(boxes, img_width, img_height, input_width, input_height):
    """
    将边界框坐标从网络输入尺寸缩放到原始图像尺寸
    
    参数:
        boxes: 边界框坐标数组
        img_width: 原始图像宽度
        img_height: 原始图像高度
        input_width: 网络输入宽度
        input_height: 网络输入高度
    返回:
        boxes: 缩放后的边界框坐标数组
    """
    # 缩放边界框到原始图像尺寸
    input_shape = np.array([input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([img_width, img_height, img_width, img_height])
    return boxes

def compute_iou(box, boxes):
    """
    计算一个边界框与多个边界框的IoU
    
    参数:
        box: 单个边界框
        boxes: 多个边界框
    返回:
        iou: IoU值数组
    """
    # 计算交集的坐标
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # 计算交集面积
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # 计算并集面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area

    return iou

def nms(boxes, scores, iou_threshold):
    """
    非极大值抑制
    
    参数:
        boxes: 边界框坐标数组
        scores: 置信度分数数组
        iou_threshold: IoU阈值
    返回:
        keep_boxes: 保留的边界框索引
    """
    # 按分数排序
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # 取分数最高的边界框
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # 计算当前边界框与其余边界框的IoU
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # 移除IoU大于阈值的边界框
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    """
    多类别非极大值抑制
    
    参数:
        boxes: 边界框坐标数组
        scores: 置信度分数数组
        class_ids: 类别ID数组
        iou_threshold: IoU阈值
    返回:
        keep_boxes: 保留的边界框索引
    """
    # 获取唯一的类别ID
    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        # 获取当前类别的边界框
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]

        # 对当前类别的边界框进行NMS
        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes

def process_output(output, img_width, img_height, input_width, input_height, 
                 conf_thres=0.5, iou_thres=0.4, scale_factor=None, pad_info=None):
    """
    处理网络输出，提取边界框、置信度和类别ID
    
    参数:
        output: 网络输出
        img_width: 原始图像宽度
        img_height: 原始图像高度
        input_width: 输入网络的图像宽度
        input_height: 输入网络的图像高度
        conf_thres: 置信度阈值
        iou_thres: IoU阈值
        scale_factor: 缩放因子
        pad_info: 填充信息 (ty1, ty2, tx1, tx2)
    返回:
        boxes: 边界框
        scores: 置信度
        class_ids: 类别ID
    """
    predictions = np.squeeze(output[0]).T

    # 过滤掉低于阈值的对象置信度分数
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_thres, :]
    scores = scores[scores > conf_thres]

    if len(scores) == 0:
        return [], [], []

    # 获取置信度最高的类别
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # 获取边界框
    boxes = predictions[:, :4]
    
    # 恢复到输入尺寸
    if scale_factor is not None and pad_info is not None:
        ty1, ty2, tx1, tx2 = pad_info
        
        # 从中心宽高转为左上右下
        boxes = xywh2xyxy(boxes)
        
        # 移除填充
        boxes[:, 0] = boxes[:, 0] - tx1
        boxes[:, 2] = boxes[:, 2] - tx1
        boxes[:, 1] = boxes[:, 1] - ty1
        boxes[:, 3] = boxes[:, 3] - ty1
        
        # 通过缩放因子恢复到原始尺寸
        boxes /= scale_factor
    else:
        # 使用旧方法进行缩放
        boxes = rescale_boxes(boxes, img_width, img_height, input_width, input_height)
        boxes = xywh2xyxy(boxes)

    # 应用非极大值抑制，抑制弱的、重叠的边界框
    indices = multiclass_nms(boxes, scores, class_ids, iou_thres)

    if len(indices) > 0:
        return boxes[indices], scores[indices], class_ids[indices]
    else:
        return [], [], [] 