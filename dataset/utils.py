from __future__ import print_function
from __future__ import division

import torchvision
from torchvision import transforms
import torchvision.transforms as T
import PIL.Image
import torch
import random
import math

def std_per_channel(images):
    images = torch.stack(images, dim = 0)
    return images.view(3, -1).std(dim = 1)


def mean_per_channel(images):
    images = torch.stack(images, dim = 0)
    return images.view(3, -1).mean(dim = 1)


class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im

class print_shape():
    def __call__(self, im):
        print(im.size)
        return im

class RGBToBGR():
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
        im = PIL.Image.merge('RGB', [b, g, r])
        return im

class pad_shorter():
    def __call__(self, im):
        h,w = im.size[-2:]
        s = max(h, w) 
        new_im = PIL.Image.new("RGB", (s, s))
        new_im.paste(im, ((s-h)//2, (s-w)//2))
        return new_im    

    
class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __oldcall__(self, tensor):
        tensor.mul_(255)
        return tensor

    def __call__(self, tensor):
        tensor = (
            tensor - self.in_range[0]
        ) / (
            self.in_range[1] - self.in_range[0]
        ) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor


def make_transform(is_train = True, is_inception = False):
    # Resolution Resize List : 256, 292, 361, 512
    # Resolution Crop List: 224, 256, 324, 448
    
    resnet_sz_resize = 256
    resnet_sz_crop = 224 
    resnet_mean = [0.485, 0.456, 0.406]
    resnet_std = [0.229, 0.224, 0.225]
    resnet_transform = transforms.Compose([
        transforms.RandomResizedCrop(resnet_sz_crop) if is_train else Identity(),
        transforms.RandomHorizontalFlip() if is_train else Identity(),
        transforms.Resize(resnet_sz_resize) if not is_train else Identity(),
        transforms.CenterCrop(resnet_sz_crop) if not is_train else Identity(),
        transforms.ToTensor(),
        transforms.Normalize(mean=resnet_mean, std=resnet_std)
    ])

    inception_sz_resize = 256
    inception_sz_crop = 224
    inception_mean = [104, 117, 128]
    inception_std = [1, 1, 1]
    inception_transform = transforms.Compose(
       [
        RGBToBGR(),
        transforms.RandomResizedCrop(inception_sz_crop) if is_train else Identity(),
        transforms.RandomHorizontalFlip() if is_train else Identity(),
        transforms.Resize(inception_sz_resize) if not is_train else Identity(),
        transforms.CenterCrop(inception_sz_crop) if not is_train else Identity(),
        transforms.ToTensor(),
        ScaleIntensities([0, 1], [0, 255]),
        transforms.Normalize(mean=inception_mean, std=inception_std)
       ])
    
    return inception_transform if is_inception else resnet_transform

def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

def build_transforms(is_train=True):
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        transform = T.Compose([
            T.Resize([384, 128]),
            T.RandomHorizontalFlip(),
            #T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(10),
            T.RandomCrop([384, 128]),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5, sl=0.02, sh=0.4, mean=[0.485, 0.456, 0.406])
        ])
    else:
        transform = T.Compose([
            T.Resize([384, 128]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform