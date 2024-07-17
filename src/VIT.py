import os
import torch.nn as nn
from transformers import ViTModel,ViTImageProcessor

class VitModel(nn.Module):
    def __init__(self,reuse:bool = True,cache_dir:str='Cache/Transformers',version:int = 1):
        super(VitModel,self).__init__()

        if reuse:
            self.model = ViTModel.from_pretrained(os.path.join('Model','VIT',f'v_{version}'))
        else:
            self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k',cache_dir=cache_dir,use_mask_token=True)

        self.classification = nn.Linear(self.model.config.hidden_size,1000)
        self.softmax = nn.Softmax(dim=1)
        self.to('cuda')

    def forward(self,image):
        image = image['pixel_values'].squeeze(1)
        x = self.model(pixel_values = image)
        x = x.pooler_output
        x = self.softmax(self.classification(x))
        return x