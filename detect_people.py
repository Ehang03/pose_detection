# 作者 xyh
# 开发日期 2025/7/21
import os
import cv2
from ultralytics import YOLO


def workouts(model_path, video_path, point_list):
    # 初始化 YOLO 模型
    model = YOLO(model_path)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("视频处理完成或无法读取帧")
            break

        # 使用 YOLO 模型进行检测和姿态估计
        results = model(im0, classes=0, conf=0.25)  # classes=0 通常表示人

        # 获取带标注的帧
        annotated_frame = results[0].plot()

        # 写入处理后的帧
        video_writer.write(annotated_frame)

        # 显示结果（可选）
        cv2.imshow("运动检测", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/俯卧撑.mp4"
    # point_list = [6, 8, 10]  # 头为正的俯卧撑，检测右肩、右肘、右手三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/俯卧撑1.mp4"
    # point_list = [5, 7, 9]  # 头为右的俯卧撑，检测左肩、左肘、左手三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/俯卧撑2.mp4"
    # point_list = [6, 8, 10]  # 头为右的俯卧撑，检测右肩、右肘、右手三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/引体向上1.mp4"
    # point_list = [6, 8, 10]  # 头为正的引体向上，检测右肩、右肘、右手三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    model_path = "./weights/yolo11x-pose.pt"
    video_path = "./video/西湖桥跳水.mp4"
    point_list = [6, 8, 10]  # 头为正的引体向上，检测右肩、右肘、右手三个点形成的夹角。
    workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/引体卷腹.mp4"
    # point_list = [6, 12, 14]  # 右侧朝向的引体卷腹，检测右肩、右腰、右膝三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/引体卷腹1.mp4"
    # point_list = [5, 11, 13]  # 左侧朝向的引体卷腹，检测左肩、左腰、左膝三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/引体卷腹2.mp4"
    # point_list = [6, 12, 14]  # 正面朝向的引体卷腹，检测右肩、右腰、右膝三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/仰卧起坐.mp4"
    # point_list = [6, 12, 14]  # 左侧朝向的仰卧起坐，检测右肩、右腰、右膝三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/V字卷腹.mp4"
    # point_list = [5, 11, 13]  # 右侧朝向的V字卷腹，检测左肩、左腰、左膝三个点形成的夹角。
    # workouts(model_path, video_path, point_list, up_angle=115)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/深蹲.mp4"
    # point_list = [11, 13, 15]  # 左侧朝向的深蹲，检测左肩、左腰、左膝三个点形成的夹角。
    # workouts(model_path, video_path, point_list)