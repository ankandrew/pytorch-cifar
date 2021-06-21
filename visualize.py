# https://github.com/jcpeterson/cifar-10h
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import model_to_class

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


# TODO: under-calibration, over-condifence metrics
class VisualizeCifar10:
    def __init__(self, model_type: str, model_path: List, test_dataloader: DataLoader):
        """
        model_type should be a name corresponding to one of
        the models of utils.model_to_class

        model_path should be a list of filenames
        """
        if model_type not in model_to_class.keys():
            raise ValueError(f'{model_type} not in model_to_class')
        self.model_type = model_type
        self.models = self.__loader_models(model_path)
        self.soft_labels_h = np.load('cifar10h-probs.npy')
        self.dataiter = iter(test_dataloader)
        self.images, self.labels = next(self.dataiter)

    def __loader_models(self, model_path: List):
        """
        Loads the models based on self.model_type and
        returns them as a dict with key/value:

        'filename': nn.Module
        """
        models = dict()
        model_filenames = [join(model_path, f) for f in listdir(model_path) if isfile(join(model_path, f))]
        for path in model_filenames:
            path = Path(path)
            if path.is_file():
                net = model_to_class[self.model_type]()
                net = torch.nn.DataParallel(net)
                checkpoint = torch.load(path)
                net.load_state_dict(checkpoint['net'])
                models[path.name] = net
            else:
                print(f'Skipping non-existent {path} file')
        return models

    @staticmethod
    def inverse_transform(img):
        """
        Restores original image
        """
        for t, m, s in zip(img, CIFAR10_MEAN, CIFAR10_STD):
            t.mul_(s).add_(m)
        return img.permute(1, 2, 0) * 255.

    def compare_distributions(self):
        """
        Visually compare predictions made by CE, LS, OLS models
        and furthermore compare CIFAR-10H human soft label distribution.
        """
        fig, axs = plt.subplots(4)
        fig.tight_layout()
        # fig.suptitle('Vertically stacked subplots')
        original_img = self.inverse_transform(self.images[0])
        original_img = original_img.int().numpy()
        axs[0].imshow(original_img)
        x = range(10)
        for i, (model_name, model) in enumerate(self.models.items(), 1):
            # Predict
            pred = model(self.images[0][None, ...]).softmax(dim=1)
            pred_class = classes[pred.argmax(dim=1).item()]
            gt_class = classes[self.labels[0]]
            axs[i].bar(classes, pred.detach().cpu().numpy()[0])
            axs[i].set_title(f'{model_name}    pred={pred_class}    gt={gt_class}')
            axs[i].set_ylim([0, 0.3])
        plt.show()


if __name__ == '__main__':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=True, num_workers=0)
    vis_cifar10 = VisualizeCifar10('resnet18', 'C:/Users/Anka/Desktop/checkpoint', testloader)
    vis_cifar10.compare_distributions()
