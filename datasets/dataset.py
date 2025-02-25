import os
import torch
import cv2
from torch.utils.data import Dataset
from albumentations import CenterCrop, Compose, Normalize, Resize
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor
from datasets.data import *

class DeepfakeDataset(Dataset):
    def __init__(self, phase='train', datalabel='', resize=(320, 320), normalize=dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])):
        assert phase in ['train', 'val', 'test']

        self.datalabel = datalabel
        self.phase = phase
        self.resize = resize
        self.normalize = normalize

        print(f"[DEBUG] Checking dataset: {self.datalabel}")
        print(f"[DEBUG] Initializing Dataset - Phase: {phase}, Label: {datalabel}")

        # Transformations
        self.trans = Compose([
            Resize(*resize),  # ✅ Resize instead of cropping
            Normalize(mean=normalize['mean'], std=normalize['std']),
            ToTensor()
        ])

        # Load dataset based on label
        self.dataset = []

        if self.datalabel == 'ff-c23':
            print("[DEBUG] Loading FF++ C23 dataset...")
            self.dataset = FF_dataset(tag='DeepFakeDetection', codec='c23', part=phase)
        elif self.datalabel == 'ff-c40':
            print("[DEBUG] Loading FF++ C40 dataset...")
            self.dataset = FF_dataset(tag='DeepFakeDetection', codec='c40', part=phase)
        elif self.datalabel == 'celebdf':
            print("[DEBUG] Loading CelebDF dataset...")
            self.dataset = Celeb_test
        elif self.datalabel == 'wilddeepfake':
            print("[DEBUG] Loading WildDeepfake dataset...")
            self.dataset = dfdc_dataset(part=phase)
        else:
            print(f"[ERROR] Unknown dataset label: {self.datalabel}")
            raise Exception(f"No such dataset: {self.datalabel}")

        # Debug: Print first 10 dataset entries
        print("[DEBUG] Dataset Loaded: First 10 Entries")
        print(self.dataset[:10])  

        if len(self.dataset) == 0:
            raise Exception("[ERROR] Dataset is empty. Check dataset paths!")

    def __getitem__(self, item):
        try:
            img_path = self.dataset[item][0]
            image = cv2.imread(img_path)
            if image is None:
                print(f"[WARNING] Failed to load image: {img_path}")
                return torch.zeros(3, self.resize[0], self.resize[1]), -1
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return self.trans(image=image)['image'], self.dataset[item][1]
        except Exception as e:
            print(f"[ERROR] Issue loading image: {e}")
            return torch.zeros(3, self.resize[0], self.resize[1]), -1

    def __len__(self):
        return len(self.dataset)
