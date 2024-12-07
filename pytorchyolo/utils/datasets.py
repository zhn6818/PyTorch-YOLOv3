from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            # 判断是否为COCO数据集
            if "coco" in image_dir.lower():
                label_dir = "labels".join(image_dir.rsplit("images", 1))
                assert label_dir != image_dir, \
                    f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            else:
                label_dir = "labels".join(image_dir.rsplit("JPEGImages", 1))
                assert label_dir != image_dir, \
                    f"Image path must contain a folder named 'JPEGImages'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)
        # self.visualize_batch(imgs, bb_targets, self.img_size)
        
        return paths, imgs, bb_targets
    
    def visualize_batch(self, imgs, bb_targets, img_size):
        """
        可视化一批图像，并在图像上绘制边界框
        :param imgs: 一个批次的图像（Tensor）
        :param bb_targets: 边界框标签（Tensor, 格式为 [batch_idx, class_id, x_center, y_center, width, height]）
        :param img_size: 图像尺寸，用于调整坐标
        """
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()  # 将图像从 (B, C, H, W) 转为 (B, H, W, C)
        bb_targets = bb_targets.cpu().numpy()         # 转为 NumPy 格式
        
        batch_size = imgs.shape[0]

        for i in range(batch_size):
            img = imgs[i] * 255.0  # 如果图像归一化了，乘以 255 恢复
            img = img.astype(np.uint8)
            img_pil = Image.fromarray(img)

            draw = ImageDraw.Draw(img_pil)

            # 获取当前图像的所有边界框
            boxes = bb_targets[bb_targets[:, 0] == i]
            for box in boxes:
                _, class_id, x_center, y_center, width, height = box

                # 将边界框中心坐标和宽高转换为左上角和右下角坐标
                x1 = (x_center - width / 2) * img_size
                y1 = (y_center - height / 2) * img_size
                x2 = (x_center + width / 2) * img_size
                y2 = (y_center + height / 2) * img_size

                # 绘制边界框
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                # 绘制类别 ID
                draw.text((x1, y1), f"{int(class_id)}", fill="yellow")

            # 显示图像
            plt.figure(figsize=(6, 6))
            plt.imshow(img_pil)
            plt.axis("off")
            plt.show()

    def __len__(self):
        return len(self.img_files)
