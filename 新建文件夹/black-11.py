import os
import shutil
import random
import cv2
import itertools
import numpy as np

def copy_folder(src, dst):
    """复制文件夹"""
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copy_folder(s, d)
        else:
            shutil.copy2(s, d)

def random_select_and_process_images(folder_path, num_images, darkness_func):
    """随机选择一定数量的图片并进行黑暗处理"""
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # 随机抽取指定数量的图片
    selected_images = random.sample(image_files, num_images)

    for image_name in selected_images:
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)  # 使用 OpenCV 读取图像
        
        # 调用黑暗处理函数
        processed_image = darkness_func(image)

        # 将处理后的图片保存回文件夹
        cv2.imwrite(image_path, processed_image)

def darkness_processing(image):
    """黑暗处理函数：对图像中的连通区域进行亮度减弱处理"""
    # 转为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 设置阈值
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # 查找连通区域（轮廓）
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 检查是否找到了足够的连通区域
    if len(contours) >= 2:  # 确保至少有两个连通区域
        # 生成所有可能的连通区域组合，这里改为随机选择一组
        combinations = list(itertools.combinations(range(len(contours)), 1))

        # 随机选择一个组合
        i = random.choice(combinations)

        # 复制原始图像
        image_copy = image.copy()

        for contour_index in i:  # 遍历组合中的轮廓
            contour = contours[contour_index]
            x, y, w, h = cv2.boundingRect(contour)  # 计算外接矩形
            # 获取该区域的图像部分
            region = image_copy[y:y+h, x:x+w]

            # 将该区域的RGB值减少50%
            region = region * 0.5

            # 确保值在有效范围内 [0, 255]，避免溢出
            region = np.clip(region, 0, 255).astype(np.uint8)

            # 将修改后的区域替换回原图
            image_copy[y:y+h, x:x+w] = (0, 0, 0)  # 将区域填充为黑色

        return image_copy
    else:
        print("未找到足够的连通区域。")
        return image

if __name__ == "__main__":
    # 文件夹路径
    source_folder = "C:/Users/DELL/Desktop/p6"
    destination_folder = "C:/Users/DELL/Desktop/b6"

    # 复制文件夹
    copy_folder(source_folder, destination_folder)

    # 随机选择并处理图片
    random_select_and_process_images(destination_folder, num_images=10, darkness_func=darkness_processing)

    # 保存视频
    wholeworld_picture_folder = "C:/Users/DELL/Desktop/b6/"
    output_path = "C:/Users/DELL/Desktop/b6.mp4"

    filelist = os.listdir(wholeworld_picture_folder)

    fps = 3
    size = (640, 480)
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, size)
    for item in filelist:
        if item.endswith('.png'):
            item = os.path.join(wholeworld_picture_folder, item)
            img = cv2.imread(item)
            video.write(img)
    video.release()
    cv2.destroyAllWindows()
