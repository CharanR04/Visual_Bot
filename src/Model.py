import os
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, BertModel, AlbertModel, AlbertTokenizer,BertTokenizer, ViTImageProcessor, GPTJModel , AutoTokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=300):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Model(nn.Module):
    def __init__(self,_model:str = 'GPT',version:int = 1,cache_dir:str = 'Cache/Transformers'):
        super(Model, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dir = 'Model'
        self._model = _model
        self.version = 'v_'+str(version)


        if os.path.exists(os.path.join(self.dir,_model, self.version)):
            self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k',cache_dir=cache_dir)
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir=cache_dir)
            self.generator_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased",cache_dir=cache_dir) if self._model == 'BERT' else AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B",cache_dir=cache_dir)
            self.generator_tokenizer.special_tokens_map['eos_token']='[EOS]'
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.load_model(version=version)
            print(sum(p.numel() for p in self.parameters() if p.requires_grad))
        else:

            self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k',cache_dir=cache_dir)
            self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k" ,cache_dir=cache_dir,use_mask_token=True).to(self.device)

            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir=cache_dir)
            self.text_encoder = AlbertModel.from_pretrained('albert-base-v2', cache_dir=cache_dir).to(self.device)
            
            if self._model == 'BERT':
                self.generator_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased",cache_dir=cache_dir)
                self.generator_tokenizer.special_tokens_map['eos_token']='[EOS]'
                self.generator = BertModel.from_pretrained('google-bert/bert-base-uncased', cache_dir=cache_dir).to(self.device)
            else:
                self.generator_tokenizer=AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B",cache_dir=cache_dir)
                self.generator_tokenizer.special_tokens_map['eos_token']='[EOS]'
                self.generator = GPTJModel.from_pretrained("EleutherAI/gpt-j-6B",cache_dir=cache_dir).to(self.device)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.pipeline =nn.Linear(768,self.generator.config.hidden_size)
            self.classifier = nn.Linear(self.generator.config.hidden_size, len(self.generator_tokenizer.vocab))

        self.positional_encoding = PositionalEncoding(dim=self.generator.config.hidden_size)

    def forward(self, input_squence,attention_mask):
        piped_out = self.relu(self.pipeline(input_squence))
        out = self.generator(inputs_embeds=piped_out, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(out)
        probabilities = F.softmax(logits[:, -1, :], dim=-1)
        return probabilities
    
    def get_sequence(self,image,text_ids):
        image_embedding = self.encode_image(image)
        text_embedding = self.encode_text(text_ids)
        merged_embedding = self.merge(image_embedding, text_embedding)
        merged_embedding = self.positional_encoding(merged_embedding)

        img_mask = torch.ones(image_embedding.size()[:-1], dtype=torch.long,device=self.device)
        txt_mask = text_ids['attention_mask'].squeeze(1)
        attention_mask = self.merge(img_mask, txt_mask)
        
        return merged_embedding,attention_mask

    def encode_image(self, image):
        outputs = self.image_encoder(pixel_values=image['pixel_values'].squeeze(1))
        embedding = outputs.last_hidden_state
        return embedding

    def encode_text(self, text_ids):
        text_ids = text_ids.to(self.device)
        attention_mask = text_ids['attention_mask'].squeeze(1)
        text_ids = text_ids['input_ids'].squeeze(1)
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=text_ids,attention_mask=attention_mask)
        embedding = outputs.last_hidden_state
        return embedding
    
    def get_text_id(self, text, max_length=50):
        text_ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding='max_length')
        return text_ids

    
    def merge(self, img_emb, text_emb):
        merged_embedding = torch.cat((img_emb, text_emb), dim=1)
        return merged_embedding
    
    def set_eval(self):
        self.image_encoder.eval()
        self.text_encoder.eval()
        self.generator.eval()
        self.pipeline.eval()
        self.classifier.eval()

    def set_train(self):
        for (name,param) in self.named_parameters():
            param.requires_grad_(True)
    
        self.image_encoder.train()
        self.generator.train()
        self.pipeline.train()
        self.classifier.train()

    def save_model(self,version:int = 1):
        version= 'v_'+str(version)
        if not os.path.exists(os.path.join(self.dir,self._model, version)):
            os.makedirs(os.path.join(self.dir,self._model, version))
        self.image_encoder.save_pretrained(os.path.join(self.dir,self._model, version,'VIT'))
        self.text_encoder.save_pretrained(os.path.join(self.dir,self._model, version,'ALBERT'))
        self.generator.save_pretrained(os.path.join(self.dir,self._model, version,self._model.upper()))
        torch.save(self.pipeline.state_dict(), os.path.join(self.dir, self._model, version, 'pipeline.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(self.dir, self._model, version, 'classifier.pth'))

    def load_model(self,version:int = 1):
        version= 'v_'+str(version)

        self.image_encoder = ViTModel.from_pretrained(os.path.join(self.dir,self._model, version,'VIT')).to(self.device)
        self.text_encoder = AlbertModel.from_pretrained(os.path.join(self.dir,self._model, version,'ALBERT')).to(self.device)
        self.generator = GPTJModel.from_pretrained(os.path.join(self.dir,self._model, version, self._model.upper())) if self._model == 'GPTJ' else BertModel.from_pretrained(os.path.join(self.dir,self._model, version,self._model.upper()))
        self.generator = self.generator.to(self.device)

        self.pipeline = nn.Linear(768, self.generator.config.hidden_size).to(self.device)
        self.pipeline.load_state_dict(torch.load(os.path.join(self.dir, self._model, version, 'pipeline.pth')))

        self.classifier = nn.Linear(self.generator.config.hidden_size, len(self.generator_tokenizer.vocab)).to(self.device)
        self.classifier.load_state_dict(torch.load(os.path.join(self.dir, self._model, version, 'classifier.pth')))