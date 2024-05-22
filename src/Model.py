import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, BertModel, AlbertModel, AlbertTokenizer,BertTokenizer, ViTImageProcessor, GPTJModel , AutoTokenizer

#generator_tokenizer=AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B",cache_dir=cache_dir)
#generator = GPTJModel.from_pretrained("EleutherAI/gpt-j-6B",cache_dir=cache_dir)

class Model(nn.Module):
    def __init__(self,cache_dir:str = 'Cache/Transformers'):
        super(Model, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k',cache_dir=cache_dir)
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k" ,cache_dir=cache_dir,use_mask_token=True).to(self.device)

        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir=cache_dir)
        self.text_encoder = AlbertModel.from_pretrained('albert-base-v2', cache_dir=cache_dir).to(self.device)
        
        self.generator_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased",cache_dir=cache_dir)
        self.generator_tokenizer.special_tokens_map['eos_token']='[EOS]'
        self.generator = BertModel.from_pretrained('google-bert/bert-base-uncased', cache_dir=cache_dir).to(self.device)
        """
        self.generator_tokenizer=AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B",cache_dir=cache_dir)
        self.generator_tokenizer.special_tokens_map['eos_token']='[EOS]'
        self.generator = GPTJModel.from_pretrained("EleutherAI/gpt-j-6B",cache_dir=cache_dir)
        """
        self.pipeline =nn.Linear(768,self.generator.config.hidden_size).to(self.device)
        self.classifier = nn.Linear(self.generator.config.hidden_size, len(self.generator_tokenizer.vocab)).to(self.device)

        self = self.to(self.device)

    def forward(self, input_squence,attention_mask):
        piped_out = self.pipeline(input_squence)
        out = self.generator(inputs_embeds=piped_out, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(out)
        probabilities = F.softmax(logits[:, -1, :], dim=-1)
        return probabilities
    
    def get_sequence(self,image,text_ids):
        image_embedding = self.encode_image(image)
        text_embedding = self.encode_text(text_ids)
        merged_embedding = self.merge(image_embedding, text_embedding)

        img_mask = torch.ones(image_embedding.size()[:-1], dtype=torch.long,device=self.device)
        txt_mask = text_ids['attention_mask'].squeeze(1)
        attention_mask = self.merge(img_mask, txt_mask)

        piped_out = self.pipeline(merged_embedding)
        return piped_out,attention_mask

    def encode_image(self, image):
        inputs = self.image_processor(images=image,do_rescale=False,return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.image_encoder(pixel_values=inputs['pixel_values'])
        embedding = outputs.last_hidden_state
        return embedding

    def encode_text(self, text_ids):
        text_ids = text_ids.to(self.device  )
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
        self.image_encoder.train()
        self.generator.train()
        self.pipeline.train()
        self.classifier.train()