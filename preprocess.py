from PIL import Image
import os

image_path = r"./dataset/TestDataset/CVC-300/images"  # 原始图像路径
image_resize_path = r"./dataset/TestDataset/CVC-300/images_resize"  # 修改后图像保存的路径
masks_path = r"./dataset/TestDataset/CVC-300/masks"  # 原始图像路径
masks_resize_path = r"./dataset/TestDataset/CVC-300/masks_resize"  # 修改后图像保存的路径

if not os.path.exists(image_resize_path):  # 如果目标文件夹不存在，则创建
    os.makedirs(image_resize_path)
if not os.path.exists(masks_resize_path):  # 如果目标文件夹不存在，则创建
    os.makedirs(masks_resize_path)

for root, dirs, files in os.walk(image_path):
    for file in files:  # 遍历每个文件
        picture_path = os.path.join(root, file)  # 生成图像的完整路径
        pic_org = Image.open(picture_path)  # 打开图像
        pic_new = pic_org.resize((384, 288), Image.LANCZOS)  # 缩放尺寸并选择
        pic_new_path = os.path.join(image_resize_path, file)  # 生成图像保存路径和文件名
        pic_new.save(pic_new_path)  # 保存修改
        print("%s 已经调完成！" % pic_new_path)  # 输出保存结果提示信息


for root, dirs, files in os.walk(masks_path):
    for file in files:  # 遍历每个文件
        picture_path = os.path.join(root, file)  # 生成图像的完整路径
        pic_org = Image.open(picture_path)  # 打开图像
        pic_new = pic_org.resize((384, 288), Image.LANCZOS)  # 缩放尺寸并选择
        pic_new = pic_new.convert("L")  # 将图像转换为灰度
        pic_new_path = os.path.join(masks_resize_path, file)  # 生成图像保存路径和文件名
        pic_new.save(pic_new_path)  # 保存修改
        print(f"{pic_new_path} 已经调完成！")  # 输出保存结果提示信息