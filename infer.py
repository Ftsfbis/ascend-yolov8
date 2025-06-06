import cv2
import time
import os
import numpy as np

from preprocessing import preprocess_image
from postprocessing import process_output
from visualization import draw_detections
from utils import save_detection_results, create_output_dir, class_names

def model_infer(image, model, input_w=640, input_h=640, conf_thres=0.5, iou_thres=0.4):
    """
    模型推理函数
    
    参数:
        image: 输入图像
        model: 推理模型
        input_w: 输入宽度
        input_h: 输入高度
        conf_thres: 置信度阈值
        iou_thres: IoU阈值
    返回:
        result_image: 绘制了检测结果的图像
        boxes: 检测框
        scores: 置信度分数
        class_ids: 类别ID
        process_time: 处理时间
    """
    start_time = time.time()
    
    # 确保输入图像是有效的
    if image is None or image.size == 0:
        print("警告：输入图像无效")
        return image, [], [], [], 0.0
    
    # 获取图像尺寸
    orig_h, orig_w, _ = image.shape
    
    try:
        # 预处理图像，获取缩放和填充信息
        img_input, _, _, _, scale_factor, pad_info = preprocess_image(
            image, input_w, input_h)
        pre_time = time.time()
        
        # 模型推理
        outputs = model.infer([img_input])
        infer_time = time.time()
        
        # 处理输出结果，传入缩放和填充信息以确保正确的坐标转换
        boxes, scores, class_ids = process_output(
            outputs, orig_w, orig_h, input_w, input_h,
            conf_thres, iou_thres, scale_factor, pad_info
        )
        post_time = time.time()
        
        # 可视化检测结果
        result_image = draw_detections(image.copy(), boxes, scores, class_ids)
        
        # 添加处理时间信息
        cv2.putText(
            result_image,
            f"Processing time: {(post_time - start_time)*1000:.4f} ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        draw_time = time.time()
        process_time = draw_time - start_time
        
    except Exception as e:
        print(f"推理过程中出错: {e}")
        return image, [], [], [], 0.0
    
    return result_image, boxes, scores, class_ids, process_time

def infer_image_advanced(img_path, model, model_path, input_w=640, input_h=640, conf_threshold=0.5, iou_threshold=0.4, output_path="runs/"):
    """
    高级图像推理函数
    
    参数:
        img_path: 图像路径
        model: 推理模型
        model_path: 模型路径
        input_w: 输入宽度
        input_h: 输入高度
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        output_path: 输出路径
    """
    # 创建输出目录
    output_dir = create_output_dir(model_path, "image", output_path)
    
    # 读取图像
    image = cv2.imread(img_path)
    if image is None:
        print(f"错误：无法读取图像 {img_path}")
        return
    
    # 模型推理
    result_image, boxes, scores, class_ids, process_time = model_infer(
        image, model, input_w, input_h, conf_threshold, iou_threshold
    )
    
    # 创建一个有名称的窗口，并调整大小
    window_name = "图像推理结果"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 根据图像大小调整窗口
    img_height, img_width = result_image.shape[:2]
    display_width = min(1280, img_width)
    display_height = min(720, img_height)
    cv2.resizeWindow(window_name, display_width, display_height)
    
    # 显示结果
    cv2.imshow(window_name, result_image)
    
    # 打印检测结果
    print(f"检测到 {len(boxes)} 个对象:")
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        class_name = class_names[class_id] if class_id < len(class_names) else "unknown"
        print(f"  {i+1}. 类别={class_name}({class_id}), 置信度={score:.4f}, 位置={box}")
    
    # 保存结果图像
    img_filename = os.path.basename(img_path)
    output_image_path = os.path.join(output_dir, f"result_{img_filename}")
    cv2.imwrite(output_image_path, result_image)
    print(f"结果图像已保存至: {output_image_path}")
    
    # 保存检测结果到文本文件
    txt_filename = os.path.splitext(img_filename)[0] + "_results.txt"
    result_file = save_detection_results(output_dir, boxes, scores, class_ids, image_name=txt_filename)
    print(f"检测结果已保存至: {result_file}")
    
    # 保存汇总信息
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"图像推理结果摘要：\n")
        f.write(f"图像路径: {img_path}\n")
        f.write(f"图像尺寸: {image.shape[1]}x{image.shape[0]}\n")
        f.write(f"总处理时间: {process_time:.4f} 秒\n")
        f.write(f"检测到的对象数量: {len(boxes)}\n\n")
        
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            class_name = class_names[class_id] if class_id < len(class_names) else "unknown"
            f.write(f"对象 {i+1}:\n")
            f.write(f"  类别: {class_name} (ID: {class_id})\n")
            f.write(f"  置信度: {score:.4f}\n")
            f.write(f"  边界框: {box}\n\n")
    
    print(f"图像推理完成。结果保存至 {output_dir}")
    # print("按任意键关闭窗口或直接关闭窗口以结束程序...")
    
    # 修改等待键盘输入的逻辑，增加窗口关闭检测
    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
        key = cv2.waitKey(100)  # 每100ms检查一次
        if key != -1:  # 如果有按键输入
            break
    
    cv2.destroyAllWindows()
    
    # 确保所有窗口都关闭 (OpenCV在某些系统上存在窗口关闭延迟的问题)
    for i in range(5):
        cv2.waitKey(1)

def infer_video_advanced(video_path, model, model_path, input_w=640, input_h=640, conf_threshold=0.5, iou_threshold=0.4, output_path="runs/"):
    """
    高级视频推理函数
    
    参数:
        video_path: 视频路径
        model: 推理模型
        model_path: 模型路径
        input_w: 输入宽度
        input_h: 输入高度
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        output_path: 输出路径
    """
    # 创建输出目录
    output_dir = create_output_dir(model_path, "video", output_path)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建输出视频
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    output_video_path = os.path.join(output_dir, f"result_{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 创建检测结果文件
    results_file = os.path.join(output_dir, f"{video_name}_detection_results.txt")
    with open(results_file, "w") as f:
        f.write("帧号,类别ID,类别名称,置信度,x1,y1,x2,y2\n")
    
    # 创建保存抽样帧的目录
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # 抽样间隔（每N帧保存一次）
    sample_interval = 30
    
    # 处理帧计数
    frame_count = 0
    total_process_time = 0
    fps_array = []
    detections_count = 0
    
    # 创建显示窗口
    window_name = "视频推理"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # 处理帧
    while True:
        ret, frame = cap.read()
        if not ret:
            # 视频结束
            # 关闭资源
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            
            # 确保所有窗口都关闭
            for i in range(5):
                cv2.waitKey(1)
                
            # 计算并保存平均FPS
            avg_fps = frame_count / total_process_time if total_process_time > 0 else 0
            with open(os.path.join(output_dir, "summary.txt"), "w") as f:
                f.write(f"视频推理结果摘要：\n")
                f.write(f"视频路径: {video_path}\n")
                f.write(f"视频尺寸: {width}x{height}\n")
                f.write(f"原始FPS: {fps:.2f}\n")
                f.write(f"处理的总帧数: {frame_count}\n")
                f.write(f"总处理时间: {total_process_time:.2f} 秒\n")
                f.write(f"平均FPS: {avg_fps:.2f}\n")
                f.write(f"检测到的对象总数: {detections_count}\n")
            
            print(f"视频推理完成。结果保存至 {output_dir}")
            print(f"- 视频文件: {output_video_path}")
            print(f"- 检测结果: {results_file}")
            print(f"- 抽样帧: {frames_dir}")
            print(f"- 总帧数: {frame_count}")
            print(f"- 平均FPS: {avg_fps:.2f}")
            
            break
        
        # 处理开始时间
        start_time = time.time()
        
        # 处理帧
        result_frame, boxes, scores, class_ids, process_time = model_infer(
            frame, model, input_w, input_h, conf_threshold, iou_threshold
        )
        
        # 更新统计信息
        total_process_time += process_time
        detections_count += len(boxes)
        
        # 计算当前FPS
        current_fps = 1.0 / process_time if process_time > 0 else 0
        fps_array.append(current_fps)
        
        # 在帧上显示FPS
        cv2.putText(
            result_frame,
            f"FPS: {current_fps:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        # 将检测结果写入文件
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(class_id)
            class_name = class_names[class_id] if class_id < len(class_names) else "unknown"
            with open(results_file, "a") as f:
                f.write(f"{frame_count},{class_id},{class_name},{score:.4f},{x1},{y1},{x2},{y2}\n")
        
        # 保存抽样帧
        if frame_count % sample_interval == 0:
            frame_output_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_output_path, result_frame)
        
        # 写入输出视频
        if result_frame is not None and result_frame.shape[0] > 0 and result_frame.shape[1] > 0:
            writer.write(result_frame)
            
            # 显示图像
            cv2.imshow(window_name, result_frame)
        else:
            print(f"警告：第{frame_count}帧结果无效，跳过显示和写入")
        
        # 处理键盘事件，按q退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 如果用户按了q，中断视频处理
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            # 确保所有窗口都关闭
            for i in range(5):
                cv2.waitKey(1)
            print("用户中断视频处理")
            break
        
        # 增加帧计数
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧 (FPS: {current_fps:.2f})")
    
    print(f"视频推理完成。结果保存至 {output_dir}") 