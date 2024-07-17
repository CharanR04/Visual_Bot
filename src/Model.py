import os
import torch
import math
from itertools import chain
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import ViTModel, BertModel, AlbertModel, AlbertTokenizer,BertTokenizer, ViTImageProcessor, GPTJModel , AutoTokenizer,GPT2Tokenizer,GPT2LMHeadModel,GPT2Config

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
    def __init__(self,_model:str = 'GPT',version:int = 1,cache_dir:str = 'Cache/Transformers',easy:bool = False,training = True):
        super(Model, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dir = 'Model'
        self._model = _model
        self.version = 'v_'+str(version)
        self.training = training

        if os.path.exists(os.path.join(self.dir,_model, self.version)):
            self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k',cache_dir=cache_dir)
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir=cache_dir)
            self.generator_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased",cache_dir=cache_dir) if self._model == 'BERT' else AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B",cache_dir=cache_dir)
            self.generator_tokenizer.special_tokens_map['eos_token']='[EOS]'
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.load_model(version=version)
        else:

            self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k',cache_dir=cache_dir)
            self.image_encoder = ViTModel.from_pretrained('Model/Vit/v_1').to(self.device)

            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir=cache_dir)
            self.text_encoder = AlbertModel.from_pretrained('albert-base-v2', cache_dir=cache_dir).to(self.device)

            if self._model == 'BERT':
                self.generator_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2",cache_dir=cache_dir)
                self.generator_tokenizer.special_tokens_map['eos_token']='[EOS]'
                configuration = GPT2Config(n_layer = 16,n_head=16,resid_pdrop = 0.8,embd_pdrop = 0.8,attn_pdrop = 0.8)
                self.generator = GPT2LMHeadModel.from_pretrained("openai-community/gpt2",cache_dir=cache_dir).to(self.device)
            else:
                self.generator_tokenizer=AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B",cache_dir=cache_dir)
                self.generator_tokenizer.special_tokens_map['eos_token']='[EOS]'
                self.generator = GPTJModel.from_pretrained("EleutherAI/gpt-j-6B",cache_dir=cache_dir).to(self.device)

            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.fc =nn.Linear(self.image_encoder.config.hidden_size,2048)
            self.pipeline = nn.Linear(2048,self.generator.config.hidden_size)
            self.fc1 = nn.Linear(self.generator.config.hidden_size, 16384)
            #self.fc2 = nn.Linear(4096, 16384)
            self.classifier = nn.Linear(16384, self.generator_tokenizer.vocab_size)
            init.xavier_uniform_(self.pipeline.weight)
            init.xavier_uniform_(self.fc1.weight)
            #init.xavier_uniform_(self.fc2.weight)
            init.xavier_uniform_(self.classifier.weight)

        self.number = (2,12) if self._model == 'BERT' else (1,28)

        self.positional_encoding = PositionalEncoding(dim=self.generator.config.hidden_size)

        self.special_token_indices = torch.tensor(self.tokenizer.all_special_ids, device=self.device)

        self.params = [[name for name,param in self.generator.named_parameters() if name.split('.')[self.number[0]]==str(i)] for i in range(self.number[1])]

        print(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self,embedding,mask):
        embedding = embedding.to(self.device)
        input_sequence = self.relu(self.fc(embedding))
        input_sequence=self.relu(self.pipeline(input_sequence))
        out = self.generator(inputs_embeds=input_sequence,attention_mask=mask)
        return out.logits
    
    def _forward(self, **values):
        if len(values)>2:
            image = values['image'].to(self.device)
            input_sequence = self.encode_image(image).requires_grad_(True)
            attention_mask = torch.ones(input_sequence.size()[:-1], dtype=torch.long, device=self.device)
            q_ids = values['q_ids']

            if q_ids:
                q_ids = q_ids.to(self.device)
                text_emb = self.encode_text(q_ids)
                text_mask = q_ids['attention_mask'].squeeze(1)
                input_sequence = torch.cat((input_sequence,text_emb),dim=1)
                attention_mask = torch.cat((attention_mask,text_mask), dim=1)
            
            input_sequence = self.relu(self.fc(input_sequence))
            input_sequence = self.relu(self.pipeline(input_sequence))

            input_sequence = torch.cat((input_sequence,values['gen_emb']),dim=1)
            attention_mask = torch.cat((attention_mask,values['gen_mask']),dim=1)

            #input_sequence = self.positional_encoding(input_sequence)
            out = self.generator(inputs_embeds=input_sequence)
            logits = out.logits
            if not self.training:
                return F.softmax(logits[:, 0, :], dim=-1)
            return logits
        else:
            #input_sequence = self.positional_encoding(values['input_sequence'])
            input_sequence = self.relu(self.fc(values['input_sequence']))
            piped_out = self.relu(self.pipeline(input_sequence))
            out = self.generator(inputs_embeds=piped_out)
            logits = out.logits
            if not self.training:
                return F.softmax(logits[:, 0, :], dim=-1)
            return  logits

    def get_sequence(self,image,text_ids):
        image_embedding = self.encode_image(image)
        text_embedding = self.encode_text(text_ids)
        merged_embedding = self.merge(image_embedding, text_embedding)

        img_mask = torch.ones(image_embedding.size()[:-1], dtype=torch.long,device=self.device)
        txt_mask = text_ids['attention_mask'].squeeze(1)
        attention_mask = self.merge(img_mask, txt_mask)
        
        return merged_embedding,attention_mask

    def encode_image(self, image):
        with torch.no_grad():
            outputs = self.image_encoder(pixel_values=image['pixel_values'].squeeze(1))
        embedding = outputs.last_hidden_state
        return embedding

    def encode_text(self, text_ids):
        text_ids = text_ids.to(self.device)
        attention_mask = text_ids['attention_mask'].squeeze(1)
        text_ids = text_ids['input_ids'].squeeze(1)
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
        param_list = list(chain.from_iterable(self.params))
        for (name,param) in self.named_parameters():
            if name not in param_list:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        self.fc.train()
        self.fc1.train()
        self.pipeline.train()
        self.classifier.train()

    def save_model(self,version:int = 1):
        version= 'v_'+str(version)
        if not os.path.exists(os.path.join(self.dir,self._model, version)):
            os.makedirs(os.path.join(self.dir,self._model, version))
        self.image_encoder.save_pretrained(os.path.join(self.dir,self._model, version,'VIT'))
        self.text_encoder.save_pretrained(os.path.join(self.dir,self._model, version,'ALBERT'))
        self.generator.save_pretrained(os.path.join(self.dir,self._model, version,self._model.upper()))
        torch.save(self.fc.state_dict(), os.path.join(self.dir, self._model, version, 'fc.pth'))
        torch.save(self.fc1.state_dict(), os.path.join(self.dir, self._model, version, 'fc1.pth'))
        torch.save(self.pipeline.state_dict(), os.path.join(self.dir, self._model, version, 'pipeline.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(self.dir, self._model, version, 'classifier.pth'))

    def load_model(self,version:int = 1):
        version= 'v_'+str(version)

        self.image_encoder = ViTModel.from_pretrained(os.path.join(self.dir,self._model, version,'VIT')).to(self.device)
        self.text_encoder = AlbertModel.from_pretrained(os.path.join(self.dir,self._model, version,'ALBERT')).to(self.device)
        self.generator = GPTJModel.from_pretrained(os.path.join(self.dir,self._model, version, self._model.upper())) if self._model == 'GPTJ' else BertModel.from_pretrained(os.path.join(self.dir,self._model, version,self._model.upper()))
        self.generator = self.generator.to(self.device)

        self.fc =nn.Linear(self.image_encoder.config.hidden_size,2048)
        self.pipeline = nn.Linear(2048,self.generator.config.hidden_size)
        self.fc1 = nn.Linear(self.generator.config.hidden_size, 4096)
        self.fc2 = nn.Linear(4096, 16384)
        self.classifier = nn.Linear(16384, len(self.generator_tokenizer.vocab))
        
        self.pipeline.load_state_dict(torch.load(os.path.join(self.dir, self._model, version, 'pipeline.pth')))
        self.classifier.load_state_dict(torch.load(os.path.join(self.dir, self._model, version, 'classifier.pth')))

    def get_next_mask(self, indices):
        is_special_token = torch.isin(indices, self.special_token_indices)
        special_token_mask = (~is_special_token).long()
        return special_token_mask.unsqueeze(1)