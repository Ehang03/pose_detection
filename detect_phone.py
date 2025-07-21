import os
import cv2
import numpy as np
from ultralytics import YOLO
import time


def calculate_angle(p1, p2, p3):
    """计算由三个点形成的夹角"""
    if None in [p1, p2, p3]:
        return None

    # 计算向量
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    # 计算角度
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos_angle)

    return np.degrees(angle)


def draw_text(image, text, position, color=(0, 255, 0), font_scale=0.8, thickness=2):
    """在图像上绘制中文文本"""
    # 计算文本框大小
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

    # 绘制文本背景
    cv2.rectangle(image,
                  (position[0] - 5, position[1] - text_size[1] - 5),
                  (position[0] + text_size[0] + 5, position[1] + 5),
                  (0, 0, 0), -1)

    # 绘制文本
    cv2.putText(image, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def workouts(model_path, video_url=None, point_list=[6, 8, 10], up_angle=130, down_angle=90, show_fps=True):
    # 初始化 YOLO 模型
    model = YOLO(model_path)

    # 打开视频流
    if video_url:
        cap = cv2.VideoCapture(video_url)
        print(f"正在连接手机摄像头: {video_url}")
    else:
        # 使用默认摄像头
        cap = cv2.VideoCapture(0)
        print("正在使用电脑摄像头进行实时姿态识别...")

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误: 无法打开摄像头或视频流")
        print("提示: 请确保手机和电脑在同一Wi-Fi网络下，并且IP地址正确")
        return

    # 获取视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

    # 动作计数相关变量
    rep_counts = {}  # 每个检测到的人对应的计数
    rep_states = {}  # 每个检测到的人对应的状态（向上/向下）
    last_angles = {}  # 每个检测到的人上次计算的角度

    # 计时相关变量
    start_time = time.time()
    frame_count = 0

    # 显示识别提示
    exercise_types = {
        (6, 8, 10): "俯卧撑（右）",
        (5, 7, 9): "俯卧撑（左）",
        (11, 13, 15): "深蹲（左）",
        (12, 14, 16): "深蹲（右）",
        (6, 12, 14): "卷腹（右）",
        (5, 11, 13): "卷腹（左）"
    }
    exercise_name = exercise_types.get(tuple(point_list), "自定义动作")

    print(f"当前识别动作: {exercise_name}")
    print(f"向上角度阈值: {up_angle}°")
    print(f"向下角度阈值: {down_angle}°")

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("无法从摄像头读取帧，可能连接已断开")
            break

        # 使用 YOLO 模型进行检测和姿态估计
        results = model(im0, classes=0, conf=0.25)  # classes=0 通常表示人

        # 获取带标注的帧
        annotated_frame = results[0].plot()

        # 在画面上显示当前识别的动作类型
        draw_text(annotated_frame, f"识别动作: {exercise_name}", (20, 90), color=(255, 165, 0))

        # 处理检测结果
        for i, result in enumerate(results[0].boxes):
            person_id = int(result.id) if result.id is not None else i
            keypoints = results[0].keypoints.xy[i].cpu().numpy()

            # 检查关键点是否有效
            valid_points = []
            for idx in point_list:
                if idx < len(keypoints) and not np.isnan(keypoints[idx][0]):
                    valid_points.append((int(keypoints[idx][0]), int(keypoints[idx][1])))
                else:
                    valid_points.append(None)

            # 如果有足够的有效点，则计算角度
            if len(valid_points) == 3 and None not in valid_points:
                angle = calculate_angle(*valid_points)

                # 初始化该人的状态
                if person_id not in rep_counts:
                    rep_counts[person_id] = 0
                    rep_states[person_id] = "down" if angle < (up_angle + down_angle) / 2 else "up"

                # 更新动作计数
                if rep_states[person_id] == "down" and angle > up_angle:
                    rep_states[person_id] = "up"
                    # 添加状态变化提示
                    draw_text(annotated_frame, "向上!", (valid_points[1][0] + 10, valid_points[1][1] - 60),
                              color=(0, 0, 255))
                elif rep_states[person_id] == "up" and angle < down_angle:
                    rep_counts[person_id] += 1
                    rep_states[person_id] = "down"
                    # 添加状态变化提示和计数提示
                    draw_text(annotated_frame, f"计数+1! 总数: {rep_counts[person_id]}",
                              (valid_points[1][0] + 10, valid_points[1][1] - 60), color=(0, 255, 0))

                last_angles[person_id] = angle

                # 在图像上绘制角度和关键点
                draw_text(annotated_frame, f"角度: {angle:.1f}°", (valid_points[1][0] + 10, valid_points[1][1] - 30))

                # 绘制连接线
                for j in range(len(valid_points) - 1):
                    if valid_points[j] and valid_points[j + 1]:
                        cv2.line(annotated_frame, valid_points[j], valid_points[j + 1], (0, 255, 0), 2)

            # 在图像上绘制人物ID和计数
            draw_text(annotated_frame, f"人物 {person_id}: 计数 {rep_counts.get(person_id, 0)}",
                      (int(result.xyxy[0][0]), int(result.xyxy[0][1]) - 10))

        # 显示总计数
        total_reps = sum(rep_counts.values())
        draw_text(annotated_frame, f"总计数: {total_reps}", (20, 30), color=(0, 0, 255), font_scale=1)

        # 显示FPS
        if show_fps:
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            draw_text(annotated_frame, f"FPS: {fps:.1f}", (20, 60), color=(255, 0, 0))

        # 写入处理后的帧
        video_writer.write(annotated_frame)

        # 显示结果
        cv2.imshow("实时姿态识别", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # 返回总计数
    return sum(rep_counts.values())


if __name__ == '__main__':
    model_path = "./weights/yolo11x-pose.pt"

    # 替换为你的手机IP地址（从IP Webcam应用中获取）
    # 格式示例："http://192.168.1.100:8080/video"
    phone_camera_url = "http://192.168.1.105:8080/video"

    # 使用手机摄像头进行实时姿态识别
    point_list = [6, 8, 10]  # 右肩、右肘、右手
    total_reps = workouts(model_path, video_url=phone_camera_url, point_list=point_list, up_angle=130, down_angle=90)

    print(f"检测到的动作次数: {total_reps}")
    print("程序已退出")