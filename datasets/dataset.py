import os
import torch
from torch.utils.data import Dataset
import cv2
from albumentations import CenterCrop, Compose
from albumentations.pytorch.transforms import ToTensor

class DeepfakeDataset(Dataset):

    # Load dataset with pre-extracted frames
    def __init__(self, phase='train', datalabel='', resize=(320, 320), normalize=dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.datalabel = datalabel  # Name of dataset
        self.dataset = []  # Stores (image path, label) pairs

        # Paths
        real_faces_path = r"D:\.THESIS\datasets\3_altered\3_1_blurring\3_1_1_original"  # ✅ Update this
        fake_faces_path = r"D:\.THESIS\datasets\3_altered\3_1_blurring\3_1_2_preprocessed"  # ✅ Update this

        # ✅ Load real images (label = 0)
        for subfolder in ["actors", "youtube"]:
            real_folder = os.path.join(real_faces_path, subfolder)
            for video in os.listdir(real_folder):  # Videos are preprocessed as frames
                video_path = os.path.join(real_folder, video)
                for frame in os.listdir(video_path):  # Load each frame
                    self.dataset.append((os.path.join(video_path, frame), 0))

        # ✅ Load fake images (label = 1)
        for subfolder in ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]:
            fake_folder = os.path.join(fake_faces_path, subfolder)
            for video in os.listdir(fake_folder):  # Videos are preprocessed as frames
                video_path = os.path.join(fake_folder, video)
                for frame in os.listdir(video_path):  # Load each frame
                    self.dataset.append((os.path.join(video_path, frame), 1))

        # Image transformations (resize + normalize)
        self.transform = Compose([CenterCrop(*resize), ToTensor(normalize=normalize)])

    # Load image and label
    def __getitem__(self, index):
        image_path, label = self.dataset[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = self.transform(image=image)['image']  # Apply transformations
        return image, label  # Return image tensor and label

    # Total number of frames
    def __len__(self):
        return len(self.dataset)
