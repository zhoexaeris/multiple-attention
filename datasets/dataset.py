import os
import torch
import cv2
from torch.utils.data import Dataset
from albumentations import CenterCrop, Compose
from albumentations.pytorch.transforms import ToTensor
from datasets.data import *  # Import dataset utilities

class DeepfakeDataset(Dataset):
    def __init__(self, phase='train', datalabel='', resize=(320, 320),
                 normalize=dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.datalabel = datalabel
        self.resize = resize
        self.normalize = normalize
        self.epoch = 0  # Keep epoch logic to shuffle frames if needed

        # Define transformations (Only Resize & Normalize)
        self.trans = Compose([
            CenterCrop(*resize),  # Apply Center Crop
            ToTensor(normalize=normalize)  # Normalize pixel values
        ])

        # Load dataset file paths
        self.dataset = self.load_dataset()

    def load_dataset(self):
        """Loads frame paths and labels for each dataset."""
        dataset = []
        
        # Define dataset root paths
        dataset_roots = {
            'ff-c23': 'datasets/ffpp/c23/',
            'ff-c40': 'datasets/ffpp/c40/',
            'celebdf': 'datasets/celebDF/',
            'wild': 'datasets/wilddeepfake/'
        }

        # Iterate through dataset types
        for dataset_name, root in dataset_roots.items():
            if dataset_name in self.datalabel:
                for class_label, class_folder in enumerate(['real', 'fake']):  # 0 for real, 1 for fake
                    class_path = os.path.join(root, class_folder)
                    if os.path.exists(class_path):
                        frame_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith(('.png', '.jpg'))]
                        dataset.extend([[fp, class_label] for fp in frame_files])

        if not dataset:
            raise Exception("No frames found! Check dataset paths.")

        return dataset

    def next_epoch(self):
        """ Shuffles dataset for the next epoch (optional) """
        self.epoch += 1
        random.shuffle(self.dataset)

    def __getitem__(self, index):
        """Load and process image frame."""
        try:
            frame_path, label = self.dataset[index]
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

            # Apply transformations (Resizing & Normalization)
            image = self.trans(image=image)['image']

            return image, label

        except Exception as e:
            print(f"Error loading frame: {e}")
            return torch.zeros(3, self.resize[0], self.resize[1]), -1  # Return blank tensor in case of failure

    def __len__(self):
        """Returns dataset size."""
        return len(self.dataset)
