from PIL import Image
import random
import numpy as np
import cv2
from osgeo import gdal
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch


def to_tensor_and_norm(imgs, labels):

    imgs = [img.transpose(1, 2, 0) for img in imgs]
    # Convert to tensor
    imgs = [TF.to_tensor(img.copy()) for img in imgs]  # 使用 .copy() 避免负步长
    labels = [torch.from_numpy(label.copy()).long().unsqueeze(dim=0) for label in labels]  # 使用 .copy() 避免负步长

    # Normalize images
    for i in range(len(imgs)):
        num_channels = imgs[i].size(0)  # 获取图像的通道数
        # 后续需要对自己数据集进行统计完善

        mean = [0.5] * num_channels  # 每个通道的均值
        std = [0.5] * num_channels  # 每个通道的标准差
        imgs[i] = TF.normalize(imgs[i], mean=mean, std=std)

    return imgs, labels


class CDDataAugmentation:
    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot=False,
            with_random_crop=False,
            with_scale_random_crop=False,
            with_random_blur=False,
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur

    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        # 随机水平翻转
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [np.flip(img, axis=2) for img in imgs]
            labels = [np.flip(label, axis=2) for label in labels]

        # 随机垂直翻转
        if self.with_random_vflip and random.random() > 0.5:
            imgs = [np.flip(img, axis=1) for img in imgs]
            labels = [np.flip(label, axis=1) for label in labels]

        # 随机旋转
        if self.with_random_rot and random.random() > 0.5:
            k = random.choice([1, 2, 3])  # 90, 180, 270 degrees
            imgs = [np.rot90(img, k, axes=(1, 2)) for img in imgs]
            labels = [np.rot90(label, k, axes=(1, 2)) for label in labels]

        # 随机裁剪
        if self.with_random_crop and random.random() > 0.5:
            crop_size = self.img_size
            h, w = imgs[0].shape[1], imgs[0].shape[2]
            if h > crop_size or w > crop_size:
                x_start = random.randint(0, w - crop_size)
                y_start = random.randint(0, h - crop_size)
                imgs = [img[:, y_start:y_start + crop_size, x_start:x_start + crop_size] for img in imgs]
                labels = [label[:, y_start:y_start + crop_size, x_start:x_start + crop_size] for label in labels]

        # 随机缩放和裁剪
        if self.with_scale_random_crop and random.random() > 0.5:
            scale_range = [1, 1.2]
            scale = random.uniform(scale_range[0], scale_range[1])
            # 缩放图像
            imgs_resized = []
            for img in imgs:
                img_resized = cv2.resize(img.transpose(1, 2, 0), None, fx=scale, fy=scale,
                                         interpolation=cv2.INTER_CUBIC)
                imgs_resized.append(img_resized.transpose(2, 0, 1))
            imgs = imgs_resized
            # 缩放标签
            labels_resized = []
            for label in labels:
                label_resized = cv2.resize(label.squeeze(), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                labels_resized.append(
                    np.array(label_resized, dtype=np.uint8).reshape(1, label_resized.shape[0], label_resized.shape[1]))
            labels = labels_resized
            # 裁剪
            crop_size = self.img_size
            h, w = imgs[0].shape[1], imgs[0].shape[2]
            if h > crop_size or w > crop_size:
                x_start = random.randint(0, w - crop_size)
                y_start = random.randint(0, h - crop_size)
                imgs = [img[:, y_start:y_start + crop_size, x_start:x_start + crop_size] for img in imgs]
                labels = [label[:, y_start:y_start + crop_size, x_start:x_start + crop_size] for label in labels]

        # 随机高斯模糊
        if self.with_random_blur and random.random() > 0.5:
            radius = random.random()
            for i in range(len(imgs)):
                img = imgs[i]
                for c in range(img.shape[0]):
                    imgs[i][c] = cv2.GaussianBlur(img[c], (5, 5), sigmaX=radius)

        # 转换为tensor并归一化
        if to_tensor:
            imgs, labels = to_tensor_and_norm(imgs, labels)

        return imgs, labels



def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype)*default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw


def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)
