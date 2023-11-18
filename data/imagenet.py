# -*- coding: utf-8 -*-
import os
import glob
import torch
import torch.utils.data as data
import numpy as np
from data.__init__ import idx_dict
from torchvision.datasets import ImageNet
from torchvision import transforms
from PIL import Image, ImageFilter
import h5py
import platform
import random
random.seed(2022)
torch.manual_seed(2022)
torch.cuda.manual_seed(2022)


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
resize_func = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])

normal_func = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

mask_trans = transforms.Compose([
        transforms.Resize((224, 224), Image.NEAREST),
])


def file_scanf2(path, contains, endswith, is_random=False, sub_ratio=1.0):
    files = glob.glob(path + '/*')
    if is_random:
        random.shuffle(files)
    input_files = []
    for f in files[:int(len(files) * sub_ratio)]:
        if platform.system().lower() == 'windows':
            f.replace('\\', '/')
        if not any([c in f.split('/')[-1] for c in contains]):
            continue
        if not f.endswith(endswith):
            continue
        input_files.append(f)
    return input_files


class ImagenetDataset(data.Dataset):
    def __init__(self, img_transfer=True):
        super(ImagenetDataset, self).__init__()
        data_path = '/path/to/Datasets/ImageNet/train'
        self.file_names = []
        for root, dirs, files in os.walk(data_path):
            for sub_dir in dirs:
                # if sub_dir != 'n01443537':
                #     continue
                imgs = file_scanf2(path=data_path + '/' + sub_dir,
                                   contains=[
                                       'n',
                                       # 'n01443537_9651', 'n01443537_3032', 'n01443537_1219', 'n01443537_6620',
                                       # 'n01443537_11018', 'n01443537_11018', 'n01443537_16422', 'n01443537_21526',
                                       # 'n01537544_2375', 'n01582220_392', 'n01614925_2400', 'n02113712_10565',
                                       # 'n02690373_4225',
                                   ],
                                   endswith='.JPEG',
                                   is_random=True, sub_ratio=0.001)
                self.file_names.extend(imgs)

        self.length = len(self.file_names)
        self.img_transfer = img_transfer
        print('Dataset length...', self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        f = self.file_names[item]
        image = Image.open(f)
        true_dir = f.split('/')[-2]
        true_idx = idx_dict[true_dir]

        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.img_transfer:
            image = resize_func(image)
            image = normal_func(image)

        return image.cuda(), true_idx


class ImagenetSegDataset(data.Dataset):
    def __init__(self, transform=None, target_transform=None):
        super(ImagenetSegDataset, self).__init__()
        data_path = '/path/to/Datasets/ImageNetS/ImageNetS50'
        img_path = data_path + '/'+'train-semi'

        self.img_names = []
        self.mask_names = []

        for root, dirs, files in os.walk(img_path):
            for sub_dir in dirs:
                # if sub_dir != 'n01443537':
                #     continue
                imgs = file_scanf2(path=img_path + '/' + sub_dir,
                                   contains=[
                                       'n',
                                       # 'n01443537_9651', 'n01443537_3032', 'n01443537_1219', 'n01443537_6620',
                                       # 'n01443537_11018', 'n01443537_11018', 'n01443537_16422', 'n01443537_21526',
                                       # 'n01537544_2375', 'n01582220_392', 'n01614925_2400', 'n02113712_10565',
                                        # 'n02690373_4225',
                                   ],
                                   endswith='.JPEG',
                                   is_random=True, sub_ratio=1)
                self.img_names.extend(imgs)

        self.length = len(self.img_names)
        print('Dataset length...', self.length)

        for name in self.img_names:
            mask_name = name.replace('train-semi', 'train-semi-segmentation').replace('JPEG', 'png')
            self.mask_names.append(mask_name)

    def __getitem__(self, index):
        image = Image.open(self.img_names[index])
        mask = Image.open(self.mask_names[index])
        mask = mask.convert('L')
        # true_dir = f.split('/')[-2]
        # true_idx = idx_dict[true_dir]

        if image.mode != 'RGB':
            image = image.convert('RGB')
        re_img = resize_func(image)
        img_tensor = normal_func(re_img)
        # input_tensor = torch.autograd.Variable(input_tensor.cuda())

        mask = np.array(mask_trans(mask)).astype('int32')
        mask[mask == 255] = -1
        mask = torch.from_numpy(mask).long()
        mask = torch.where(mask > 0, 1, 0)

        return img_tensor, mask

    def __len__(self):
        return len(self.img_names)


class ImageNet_blur(ImageNet):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        gauss_blur = ImageFilter.GaussianBlur(11)
        median_blur = ImageFilter.MedianFilter(11)

        blurred_img1 = sample.filter(gauss_blur)
        blurred_img2 = sample.filter(median_blur)
        blurred_img = Image.blend(blurred_img1, blurred_img2, 0.5)

        if self.transform is not None:
            sample = self.transform(sample)
            blurred_img = self.transform(blurred_img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, blurred_img), target


class Imagenet_Segmentation(data.Dataset):
    CLASSES = 2

    def __init__(self,
                 path,
                 transform=None,
                 target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        # self.h5py = h5py.File(path, 'r+')
        self.h5py = None
        tmp = h5py.File(path, 'r')
        self.data_length = len(tmp['/value/img'])
        tmp.close()
        del tmp

    def __getitem__(self, index):

        if self.h5py is None:
            self.h5py = h5py.File(self.path, 'r')

        img = np.array(self.h5py[self.h5py['/value/img'][index, 0]]).transpose((2, 1, 0))
        target = np.array(self.h5py[self.h5py[self.h5py['/value/gt'][index, 0]][0, 0]]).transpose((1, 0))

        img = Image.fromarray(img).convert('RGB')
        target = Image.fromarray(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = np.array(self.target_transform(target)).astype('int32')
            target = torch.from_numpy(target).long()

        return img, target

    def __len__(self):
        # return len(self.h5py['/value/img'])
        return self.data_length


class Imagenet_Segmentation_Blur(data.Dataset):
    CLASSES = 2

    def __init__(self,
                 path,
                 transform=None,
                 target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        # self.h5py = h5py.File(path, 'r+')
        self.h5py = None
        tmp = h5py.File(path, 'r')
        self.data_length = len(tmp['/value/img'])
        tmp.close()
        del tmp

    def __getitem__(self, index):

        if self.h5py is None:
            self.h5py = h5py.File(self.path, 'r')

        img = np.array(self.h5py[self.h5py['/value/img'][index, 0]]).transpose((2, 1, 0))
        target = np.array(self.h5py[self.h5py[self.h5py['/value/gt'][index, 0]][0, 0]]).transpose((1, 0))

        img = Image.fromarray(img).convert('RGB')
        target = Image.fromarray(target)

        gauss_blur = ImageFilter.GaussianBlur(11)
        median_blur = ImageFilter.MedianFilter(11)

        blurred_img1 = img.filter(gauss_blur)
        blurred_img2 = img.filter(median_blur)
        blurred_img = Image.blend(blurred_img1, blurred_img2, 0.5)

        # blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
        # blurred_img2 = np.float32(cv2.medianBlur(img, 11))
        # blurred_img = (blurred_img1 + blurred_img2) / 2

        if self.transform is not None:
            img = self.transform(img)
            blurred_img = self.transform(blurred_img)

        if self.target_transform is not None:
            target = np.array(self.target_transform(target)).astype('int32')
            target = torch.from_numpy(target).long()

        return (img, blurred_img), target

    def __len__(self):
        # return len(self.h5py['/value/img'])
        return self.data_length


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from tqdm import tqdm
    from imageio import imsave

    # Data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_img_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    test_lbl_trans = transforms.Compose([
        transforms.Resize((224, 224), Image.NEAREST),
    ])

    ds = Imagenet_Segmentation('/path/to/Datasets/imagenet-seg/other/gtsegs_ijcv.mat',
                               transform=test_img_trans, target_transform=test_lbl_trans)

    for i, (img, tgt) in enumerate(tqdm(ds)):
        tgt = (tgt.numpy() * 255).astype(np.uint8)
        imsave('/path/to/imagenet/gt/{}.png'.format(i), tgt)

    print('here')