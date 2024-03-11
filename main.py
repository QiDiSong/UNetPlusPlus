import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from model_construct import UNetPlusPlus as UNetPlusPlus


# Define your Dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, target_paths, transform=None, transform_label=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform
        self.transform_label = transform_label

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.transform_label is not None:
            mask = self.transform_label(mask)
        return image, mask

    def __len__(self):
        return len(self.image_paths)


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

# Listing the image and mask paths
image_paths = [os.path.join(image_dir, img_name) for img_name in sorted(os.listdir(image_dir))]
mask_paths = [os.path.join(mask_dir, mask_name) for mask_name in sorted(os.listdir(mask_dir))]

# Assuming your images are loaded and split into image_paths and mask_paths
train_dataset = CustomDataset(image_paths, mask_paths, transform, transform_label)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)


# Define Dice coefficient and loss6
def dice_coeff(pred, target):
    smooth = 1e-6
    # Flatten
    pred_probs = torch.sigmoid(pred)  # 应用 sigmoid 函数
    pred_flat = pred_probs.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        return 1 - dice_coeff(input, target)


if __name__ == '__main__':
    all_zero_num = 0
    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Create an instance of the model
    model = UNetPlusPlus(in_channels=3, out_channels=1).to(device)  # 假设是一个3通道输入和1通道输出的分割任务
    # 打印模型架构
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceLoss()
    # Training loop
    num_epochs = 100
    loss_best = 10000
    epoch = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("Epoch: ", epoch + 1, " Index: ", i, " Mean Loss: ", epoch_loss / (i + 1), "Dice系数: ", 1 - loss.item())

        # Calculate the average loss
        train_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')
        PATH = './unet++.pth'
        if train_loss < loss_best:
            loss_best = train_loss
            torch.save(model.state_dict(), "./unet++_best.pth")
        torch.save(model.state_dict(), PATH)
