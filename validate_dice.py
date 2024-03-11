import os
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision import transforms
from main import CustomDataset, dice_coeff
from model_construct import UNetPlusPlus as UNetPlusPlus


def paint(preds, title):
    import matplotlib.pyplot as plt
    # 假设 preds 是一个批量的二值化预测，这里只画出批量中的第一张
    pred_to_plot = preds[0].squeeze()  # 移除批次维度，并假设它是单通道的
    # 将张量转换为 numpy 数组以便绘图
    pred_to_plot = pred_to_plot.cpu().numpy()
    # 使用 matplotlib 画出图像
    plt.imshow(pred_to_plot, cmap='gray')  # 用灰阶的颜色映射来画出二值化的预测
    plt.title(title)
    plt.axis('off')  # 关闭坐标轴
    plt.show()


def plot_image(image_tensor):
    """
    Plot a single image tensor using matplotlib.

    Parameters:
    - image_tensor: a torch.Tensor, which is the image data
    """
    # 检查是否有批次维度，并移除它
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor.squeeze(0)

    # 检查通道维度并调整为最后一维
    if image_tensor.shape[0] == 3:
        # 如果是三通道图像，调整通道顺序为(H,W,C)
        image_tensor = image_tensor.permute(1, 2, 0)

    # 将张量转换为numpy数组
    image_numpy = image_tensor.cpu().detach().numpy()
    # 如果图像数据在[0,1]之外，则重新归一化到[0,1]
    if image_numpy.min() < 0 or image_numpy.max() > 1:
        image_numpy = (image_numpy - image_numpy.min()) / (image_numpy.max() - image_numpy.min())

    # 绘图
    plt.imshow(image_numpy)
    plt.title('Image')
    plt.axis('off')  # 关闭坐标轴
    plt.show()


# 请在这里定义您的验证数据加载器
# Data preprocessing
# Convert your data into a PyTorch dataset and create your DataLoader
transform = transforms.Compose([
    transforms.Resize((288, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_label = transforms.Compose([
    transforms.Resize((288, 384)),
    transforms.ToTensor(),
])


# Dataset paths
image_dir = "./data/image"
mask_dir = "./data/masks"
# image_dir = "./dataset/TrainDataset/images_new"               # Dice: 0.9014
# mask_dir = "./dataset/TrainDataset/masks_new"                 # 你也可以用这个数据集去验证模型的Dice值，这个数据量大，比较准确
image_dir = "./dataset/TestDataset/CVC-ClinicDB/images"         # Dice: 0.8817
mask_dir = "./dataset/TestDataset/CVC-ClinicDB/masks"
# image_dir = "./dataset/TestDataset/CVC-300/images_resize"     # 这个是修改CVC-300图像之后尺寸为(288*384)后的路径
# mask_dir = "./dataset/TestDataset/CVC-300/masks_resize"

# Listing the image and mask paths
val_image_paths = [os.path.join(image_dir, img_name) for img_name in sorted(os.listdir(image_dir))]
val_mask_paths = [os.path.join(mask_dir, mask_name) for mask_name in sorted(os.listdir(mask_dir))]

val_dataset = CustomDataset(val_image_paths, val_mask_paths, transform, transform_label)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 初始化用于跟踪最佳Dice系数的变量
best_dice = 0.0
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = UNetPlusPlus(in_channels=3, out_channels=1).to(device)  # 假设是一个3通道输入和1通道输出的分割任务
model.load_state_dict(torch.load("./unet++_best.pth"))          # 加载训练好的模型

for epoch in range(num_epochs):
    # 验证步骤
    count_masks = len(val_loader)
    with torch.no_grad():  # 在验证期间不计算梯度
        val_dice = 0.0
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            # 在这里加一些打印图片的代码
            # 画出来masks和output
            plot_image(images)                           # 画原始图
            paint(masks, "Masks")                   # 画数据集中mask图
            paint(outputs, "Model Prediction")      # 画模型推理结果
            # 计算Dice系数
            dice_val = dice_coeff(outputs, masks)
            if dice_val < 0.5:
                count_masks = count_masks - 1
            else:
                val_dice += dice_val.item()
                print("Dice系数: ", dice_val.item())

        # 计算平均Dice系数
        val_dice /= count_masks
        print("count masks: ", count_masks, "Total masks: ", len(val_loader))
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Dice: {val_dice:.4f}')
