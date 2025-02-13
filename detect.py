import cv2
import numpy as np
import os

class ImageDetector:
    def __init__(self, image_folder, output_folder, threshold=(40, 40, 40), area_threshold=10):
        """
        初始化 ImageDetector 类，设置图像文件夹路径、输出文件夹路径及检测参数。
        
        :param image_folder: 存放图像的文件夹路径
        :param output_folder: 保存结果图像的文件夹路径
        :param threshold: 用于生成掩膜的RGB阈值，像素值大于此阈值的区域被认为是前景
        :param area_threshold: 连通区域的最小面积阈值，只有大于该阈值的区域才会被考虑
        """
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.threshold = threshold
        self.area_threshold = area_threshold

    def detect(self, frame_id):
        """
        检测给定图像中的连通区域，返回每个连通区域的边界框和平均RGB值，并保存图像到指定文件夹。
        
        :param frame_id: 帧的ID，用于选择相应的图像
        :return: 返回包含每个连通区域信息的列表，每个元素是一个字典，包含 {x1, y1, x2, y2, avg_rgb}
        """
        # 根据frame_id构建图像路径，读取图像
        image_name = f"{frame_id:04d}.png"  # 使用4位数字命名格式
        image_path = os.path.join(self.image_folder, image_name)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"错误：无法读取图像 {image_path}")
            return []

        # 创建掩膜，选取符合阈值的像素作为前景
        mask = (frame[:, :, 0] > self.threshold[0]) & (frame[:, :, 1] > self.threshold[1]) & (frame[:, :, 2] > self.threshold[2])
        binary_mask = np.zeros_like(frame[:, :, 0])
        binary_mask[mask] = 255

        # 找到所有连通区域
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections ={}  # 存储检测到的区域信息
        if frame_id not in detections:
            detections[frame_id] = []
        
        # 遍历每一个连通区域
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)  # 计算区域的面积
            if area < self.area_threshold:
                continue  # 跳过面积小于阈值的区域

            # 计算区域的边界框
            x, y, w, h = cv2.boundingRect(contour)
            top_left = (x, y)
            bottom_right = (x + w, y + h)

            # 计算区域的中心点
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # 计算区域内的平均RGB值
            roi = frame[y:y+h, x:x+w]
            avg_rgb = np.mean(roi, axis=(0, 1))  # 对区域内所有像素进行平均，计算平均RGB值
            avg_rgb = tuple(map(int, avg_rgb))  # 将RGB值转换为整数
            avg_rgb = np.mean(avg_rgb)  # 将RGB值转换为整数

            # 将检测到的区域信息保存到列表中
            detections[frame_id].append([ x - 20,y - 20,x + w + 20,y + h + 20,1,avg_rgb])

            # 在图像上绘制边界框和平均RGB值
            cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 2)  # 绘制矩形框
            cv2.putText(frame, f"ID: {idx}, RGB: {avg_rgb}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 显示ID和平均RGB

        # 创建输出文件夹，如果不存在的话
        os.makedirs(self.output_folder, exist_ok=True)

        # 保存带标注的图像到输出文件夹
        output_image_path = os.path.join(self.output_folder, f"frame_{frame_id:04d}_result.png")
        cv2.imwrite(output_image_path, frame)

        print(f"图像已保存到: {output_image_path}")
        
        return detections

# 示例用法：
if __name__ == "__main__":
    # 设置图像文件夹路径、输出文件夹路径
    image_folder = "C:/Users/DELL/Desktop/a1"  # 图像文件夹路径
    output_folder = "C:/Users/DELL/Desktop/output"  # 保存标注结果的文件夹路径

    # 创建 ImageDetector 类的实例
    detector = ImageDetector(image_folder, output_folder, threshold=(40, 40, 40), area_threshold=10)

    # 选择帧号进行检测
    frame_id = 2  # 选择的帧号，图像文件名应为"0002.png"

    # 调用 detect 方法进行检测
    detections = detector.detect(frame_id)


        # 输出每个区域的边界框和平均RGB值
    for detection in detections[frame_id]:
        print(f"{detection[0]},{detection[1]},{detection[2]},{detection[3]},{detection[4]}")

