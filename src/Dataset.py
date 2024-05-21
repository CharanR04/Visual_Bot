import torch
import json
from src.API import VizWiz
from torch.utils.data import Dataset
from PIL import Image
from src.Model import Model
from transformers import AlbertModel, AlbertTokenizer
from torchvision import transforms

class CaptionDataset(Dataset):
    def __init__(self, captions_file:str, cache_dir:str = 'Cache/Transformers'):
        self.dataset = VizWiz(annotation_file = captions_file)
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir=cache_dir)
        self.text_encoder = AlbertModel.from_pretrained('albert-base-v2', cache_dir=cache_dir)
        self.text_encoder.eval()
        self.transfrom = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        if idx not in self.dataset.anns:
            return self.__getitem__(idx-1)
        ann = self.dataset.loadAnns([idx])
        
        img_id = ann[0]['image_id']
        img_desc = self.dataset.loadImgs([img_id])[0]
        img_file = img_desc['file_name']
        img_path = f'Data/Descriptive/train/{img_file}'
        img = Image.open(img_path).convert('RGB')
        
        txt = ann[0]['caption']
        txt_enc = self.encode_text(txt)

        img = self.transfrom(img)

        return img.to(self.device),txt_enc.to(self.device)
    
    def encode_text(self, text,max_length = 50):
        text_ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding = 'max_length')
        return text_ids
    
class QADataset(Dataset):
    def __init__(self, qa_file: str, cache_dir: str = 'Cache/Transformers'):
        # Load dataset from JSON file
        with open(qa_file, 'r') as file:
            self.dataset = json.load(file)
        
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir=cache_dir)
        self.text_encoder = AlbertModel.from_pretrained('albert-base-v2', cache_dir=cache_dir)
        self.text_encoder.eval()
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.data_points = []
        for idx in range(len(self.dataset)):
            img_path = f'Data/Descriptive/train/{self.dataset[idx]["image"]}'
            question = self.dataset[idx]["question"]
            answers = [ans["answer"] for ans in self.dataset[idx]["answers"] if ans["answer_confidence"] == 'yes']
            for answer in answers:
                self.data_points.append((img_path, question, answer))

    def __len__(self):
        return len(self.data_points)
    
    def get_QAimage(self, img_path, question):
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        que_ids = self.encode_text(question)
        
        return img, que_ids
    
    def encode_text(self, text, max_length=50):
        text_ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding='max_length')
        return text_ids
    
    def __getitem__(self, index):
        img_path, question, answer = self.data_points[index]
        img, que_ids = self.get_QAimage(img_path, question)
        answer_ids = self.encode_text(answer)
        
        return (img.to(self.device), que_ids.to(self.device)), answer_ids.to(self.device)