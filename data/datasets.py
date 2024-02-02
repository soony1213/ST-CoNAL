import torchvision.transforms as transforms
import os

def tiny_imagenet():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    if os.path.isdir("../scratch/datasets/tiny-imagenet/by-image/"):
        datadir = "../scratch/datasets/tiny-imagenet/by-image/"
    elif os.path.isdir("../data-local/tiny-imagenet/by-image"):
        datadir = "../data-local/tiny-imagenet/by-image"
    print("Using TinyImagenet from", datadir)

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': datadir,
        'num_classes': 200
    }


def caltech256():
    import numpy as np
    from PIL import Image, ImageEnhance
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229,  0.224,  0.225])
    # enhancers = {
    #     0: lambda image, f: ImageEnhance.Color(image).enhance(f),
    #     1: lambda image, f: ImageEnhance.Contrast(image).enhance(f),
    #     2: lambda image, f: ImageEnhance.Brightness(image).enhance(f),
    #     3: lambda image, f: ImageEnhance.Sharpness(image).enhance(f)
    # }
    # factors = {
    #     0: lambda: np.random.normal(1.0, 0.3),
    #     1: lambda: np.random.normal(1.0, 0.1),
    #     2: lambda: np.random.normal(1.0, 0.1),
    #     3: lambda: np.random.normal(1.0, 0.3),
    # }
    # def enhance(image):
    #     order = [0, 1, 2, 3]
    #     np.random.shuffle(order)
    #     for i in order:
    #         f = factors[i]()
    #         image = enhancers[i](image, f)
    #     return image

    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.Lambda(enhance),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    if os.path.isdir("../scratch/datasets/caltech256/by-image/"):
        datadir = "../scratch/datasets/caltech256/by-image/"
    elif os.path.isdir("../data-local/images/caltech/caltech256/by-image"):
        datadir = "../data-local/images/caltech/caltech256/by-image"
    print("Using CALTECH256 from", datadir)

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': datadir,
        'num_classes': 256
    }

