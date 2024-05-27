import torch
import os
from tqdm import tqdm
import json
from src.API import VizWiz
from torch.utils.data import Dataset
from PIL import Image
from src.Model import Model
from transformers import AlbertModel, AlbertTokenizer, ViTImageProcessor
from torchvision import transforms

class CaptionDataset(Dataset):
    def __init__(self, captions_file:str, cache_dir:str = 'Cache/Transformers',small:bool= False):
        self.dataset = VizWiz(annotation_file = captions_file)
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir=cache_dir)
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k',cache_dir=cache_dir)
        self.small = small
        #self.mean,self.std = self.compute_mean_and_std()
        #self.transfrom = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=self.mean, std=self.std)])
        self.transfrom = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])

    def __len__(self):
        return len(self.dataset)//32 if self.small else len(self.dataset)

    def __getitem__(self, idx):

        if idx not in self.dataset.anns:
            return self.__getitem__(idx-1)
        ann = self.dataset.loadAnns([idx])
        
        img_id = ann[0]['image_id']
        img_desc = self.dataset.loadImgs([img_id])[0]
        img_file = img_desc['file_name']
        img_path = f'Data/Images/train/{img_file}'
        img = Image.open(img_path).convert('RGB')
        
        txt = ann[0]['caption']
        txt_enc = self.encode_text(txt)

        img = self.transfrom(img)
        img = self.image_processor(images=img,do_rescale=False,return_tensors="pt")

        return img,txt_enc
    
    def encode_text(self, text,max_length = 50):
        text_ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding = 'max_length')
        return text_ids
    
    def compute_mean_and_std(self):
        cache_file = 'mean_and_std.pt'
        if os.path.exists(cache_file):
            print('Reusing cached mean and std')
            d = torch.load(cache_file)
            return d['mean'], d['std']
        return torch.tensor([0.485, 0.456, 0.406]),torch.tensor([0.229, 0.224, 0.225])

    
class QADataset(Dataset):
    def __init__(self, qa_file: str, cache_dir: str = 'Cache/Transformers',small :bool = False):
        # Load dataset from JSON file
        with open(qa_file, 'r') as file:
            self.dataset = json.load(file)
        
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir=cache_dir)
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k',cache_dir=cache_dir)
        self.small = small
        self.mean,self.std = self.compute_mean_and_std()
        self.transfrom = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=self.mean, std=self.std)])
        self.transfrom = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])

        self.data_points = []
        for idx in range(len(self.dataset)):
            img_path = f'Data/Images/train/{self.dataset[idx]["image"]}'
            question = self.dataset[idx]["question"]
            answers = [ans["answer"] for ans in self.dataset[idx]["answers"] if ans["answer_confidence"] == 'yes']
            for answer in answers:
                self.data_points.append((img_path, question, answer))

    def __len__(self):
        return len(self.data_points)//128 if self.small else len(self.data_points)
    
    def get_QAimage(self, img_path, question):
        img = Image.open(img_path).convert('RGB')
        img = self.transfrom(img)
        
        que_ids = self.encode_text(question)
    
        return img, que_ids
    
    def encode_text(self, text, max_length=50):
        text_ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding='max_length')
        return text_ids
    
    def __getitem__(self, index):
        img_path, question, answer = self.data_points[index]
        img, que_ids = self.get_QAimage(img_path, question)
        answer_ids = self.encode_text(answer)
        img = self.image_processor(images=img,do_rescale=False,return_tensors="pt")
        
        return (img, que_ids), answer_ids
    
    def compute_mean_and_std(self):
        cache_file = 'mean_and_std.pt'
        if os.path.exists(cache_file):
            print('Reusing cached mean and std')
            d = torch.load(cache_file)

            return (d['mean'], d['std'])
        return torch.tensor([0.4022, 0.4041, 0.4057]),torch.tensor([0.1338, 0.1342, 0.1351])