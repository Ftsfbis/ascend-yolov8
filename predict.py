import os
import sys
import time
from ais_bench.infer.interface import InferSession

# 导入自定义模块
from utils import class_names
from infer import infer_image_advanced, infer_video_advanced

# 模型和输入输出相关配置
model_path = 'models/yolov8s.om'
# model_path = 'models/yolov8s-p2-cbam.om'
IMG_PATH = 'images/0000006_00159_d_0000001.jpg'
VIDEO_PATH = 'videos/video3.mp4'  # 视频文件路径，根据需要修改

OUTPUT_PATH = 'runs/'

# 推理参数
conf_threshold = 0.5
iou_threshold = 0.4
input_w = 640
input_h = 640

if __name__ == "__main__":
    try:
        # 初始化推理模型
        model = InferSession(0, model_path)
        
        # 选择处理图像还是视频
        if len(sys.argv) > 1:
            if sys.argv[1] == "video":
                print("开始视频推理...")
                infer_video_advanced(
                    VIDEO_PATH, 
                    model, 
                    model_path, 
                    input_w, 
                    input_h, 
                    conf_threshold, 
                    iou_threshold, 
                    OUTPUT_PATH
                )
            elif sys.argv[1] == "image":
                print("开始图像推理...")
                infer_image_advanced(
                    IMG_PATH, 
                    model, 
                    model_path, 
                    input_w, 
                    input_h, 
                    conf_threshold, 
                    iou_threshold, 
                    OUTPUT_PATH
                )
            else:
                print(f"未知的推理模式: {sys.argv[1]}")
                print("使用方法: python predict.py [image|video]")
        else:
            print("开始图像推理...")
            infer_image_advanced(
                IMG_PATH, 
                model, 
                model_path, 
                input_w, 
                input_h, 
                conf_threshold, 
                iou_threshold, 
                OUTPUT_PATH
            )
            
        print("------------------")
        print("推理完成")
    except KeyboardInterrupt:
        print("用户中断程序")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 确保所有OpenCV窗口都关闭
        import cv2
        cv2.destroyAllWindows()
        print("程序正常退出")