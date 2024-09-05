import torch
# import glob
import os
from PIL import Image
# import cv2
# from utils.Helpers import Helpers
# import numpy as np
from utils.utilities import getData, getClassesNames

# from pycocotools.coco import COCO

class HARDataset(torch.utils.data.Dataset):
    """ HARDataset"""

    def __init__(self, train_path, test_path,subset="train", seed=251):

        assert subset=="train" or subset=="test"

        classesNames = getClassesNames(train_path,test_path)
        
        images_path = ""
        if subset=="train":
            images_path = train_path
        else:
            images_path = test_path
        
        
        images_paths, image_labels, classes_idnames, classes_nameids = getData(images_path=images_path,
                                                                               classes_names=classesNames)
        
        self.classes_idnames = classes_idnames
        self.classes_nameids = classes_nameids
        self.labels = image_labels
        self.images = images_paths

        # self.helpers = Helpers()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target)
        """
        image, label = None, None

        if os.path.exists(self.images[index]):
            # self.create_one_hot_encoding(self.labels[index])
            image = Image.open(self.images[index])
            label = self.classes_nameids[self.labels[index]]

        return image, label