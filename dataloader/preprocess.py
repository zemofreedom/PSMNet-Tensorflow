#import torch
#import torchvision.transforms as transforms
import random
import numpy as np

#数据增强

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

#__imagenet_stats = {'mean': [0.5, 0.5, 0.5],
#                   'std': [0.5, 0.5, 0.5]}

__imagenet_pca = {
    'eigval': np.array([0.2175, 0.0188, 0.0045]),
    'eigvec': np.array([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


#transforms.Compose(t_list)表示对t_list循环所有的transform操作

#图像归一化
def scale_crop(input_img, normalize=__imagenet_stats):
    #首先将图像变换到[0,1]之间
    w,h = input_img.size
    input_img = np.array(input_img,dtype=np.float32)/255
    #Normalized_image=(image-mean)/std,注意是RGB
    input_img[:,:,0] = (input_img[:,:,0] - __imagenet_stats['mean'][0])/__imagenet_stats['std'][0]
    input_img[:,:,1] = (input_img[:,:,1] - __imagenet_stats['mean'][1])/__imagenet_stats['std'][1]
    input_img[:,:,2] = (input_img[:,:,2] - __imagenet_stats['mean'][2])/__imagenet_stats['std'][2]
    #print(input_img.shape)
    #返回值为一个(h,w,3)的矩阵
    return input_img


"""
#图像随机裁剪然后归一化
def scale_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Scale(scale_size)] + t_list

    transforms.Compose(t_list)

#图像随机裁剪，随机水平翻转，归一化
def pad_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    padding = int((scale_size - input_size) / 2)
    return transforms.Compose([
        transforms.RandomCrop(input_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])


#图像随机尺寸裁剪，随机水平翻转，归一化
def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])

#改变图像的亮度，对比度和饱和度，然后归一化
def inception_color_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        #transforms.RandomSizedCrop(input_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(**normalize)
    ])
"""


"""

class Lighting(object):
    #Lighting noise(AlexNet - style PCA - based noise)

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

#讲输入图像变成灰度
class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
     #Composes several transforms together in random order.
    

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))
"""
if __name__ == '__main__':
    from PIL import Image, ImageOps
    img = Image.open('./000000_10.png').convert('RGB')
    print(img.size)
    print(np.array(img).shape)
    scale_crop(img, normalize=__imagenet_stats)
