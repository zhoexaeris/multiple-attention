import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from albumentations import CenterCrop, Compose
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor
import cv2
from datasets.data import *

class DeepfakeDataset(Dataset):
    def __init__(self, phase='train', datalabel='', resize=(320, 320), normalize=dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])):
        assert phase in ['train', 'val', 'test']
        
        # Dataset and phase setup
        self.datalabel = datalabel
        self.phase = phase
        self.resize = resize
        self.normalize = normalize

        # Define transformations
        self.trans = Compose([CenterCrop(*resize), ToTensor(normalize=normalize)])

        # Initialize the dataset based on datalabel
        self.dataset = []
        if 'ff-5' in self.datalabel:
            for i, j in enumerate(['Origin', 'Deepfakes', 'NeuralTextures', 'FaceSwap', 'Face2Face']):
                temp = FF_dataset(j, self.datalabel.split('-')[2], phase)
                temp = [[k[0], i] for k in temp]
                self.dataset += temp
        elif 'ff-all' in self.datalabel:
            for i in ['Origin', 'Deepfakes', 'NeuralTextures', 'FaceSwap', 'Face2Face']:
                self.dataset += FF_dataset(i, self.datalabel.split('-')[2], phase)
        elif 'celeb' in self.datalabel:
            self.dataset = Celeb_test
        elif 'deeper' in self.datalabel:
            self.dataset = deeperforensics_dataset(phase) + FF_dataset('Origin', self.datalabel.split('-')[1], phase)
        elif 'dfdc' in self.datalabel:
            self.dataset = dfdc_dataset(phase)
        else:
            raise Exception('No such dataset')

    def __getitem__(self, item):
        try:
            # Load image (frame) from dataset
            img_path = self.dataset[item][0]  # Directly fetch the image path
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transformation (resize and normalization)
            return self.trans(image=image)['image'], self.dataset[item][1]  # Returning the image and its label
        except Exception as e:
            print(f"Error loading image: {e}")
            return torch.zeros(3, self.resize[0], self.resize[1]), -1  # Return a dummy image for error cases

    def __len__(self):
        # Number of frames in the dataset
        return len(self.dataset)
