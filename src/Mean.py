import torch
import os
from tqdm import tqdm
import json
from PIL import Image
from torchvision import transforms

class ImagesDataset:
    def __init__(self,img_file:str='Data/Annotations/QA/train.json'):
        with open(img_file, 'r') as file:
            self.dataset = json.load(file)

    def __len__(self):
        return (len(self.dataset))
    
    def __getitem__(self, idx):
        image_file = self.dataset[idx]["image"]
        img = Image.open(f'Data/Images/train/{image_file}').convert('RGB')
        return img
    
    def compute_mean_and_std(self):
        cache_file = 'mean_and_std.pt'
        if os.path.exists(cache_file):
            print('Reusing cached mean and std')
            d = torch.load(cache_file)

            return (d['mean'], d['std'])
        transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])

        mean = torch.zeros(3)
        std = torch.zeros(3)
        for i in tqdm(range(len(self)), total=len(self), ncols=60):
            img= self[i]
            img = transform(img)
            mean += img.mean(dim=(1, 2))
            std += img.std(dim=(1, 2))

        mean /= i+1
        std /= i+1


        torch.save({'mean': mean, 'std': std}, cache_file)

        return (mean, std)