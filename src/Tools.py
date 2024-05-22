import torch
import os
from tqdm import tqdm
from src.API import VizWiz
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
from src.Dataset import QADataset,CaptionDataset

def update_sequence_mask(model,probabilities, input_sequence, attention_mask):
    _, next_token = torch.max(probabilities, dim=-1)
    next_token_embedding = model.generator.get_input_embeddings()(next_token)
    input_sequence = torch.cat((input_sequence, next_token_embedding.unsqueeze(1)), dim=1)

    new_attention_mask = torch.ones((attention_mask.size(0), 1), dtype=torch.int,device=model.device)
    attention_mask = torch.cat((attention_mask, new_attention_mask), dim=1)

    return input_sequence,attention_mask

def generate_text(model, image, question:str=None, max_length:int=50):
    model.set_eval()
    eos_embedding,eos_attention_mask = get_eos_embedding(model,1)
    
    if question:
        q_ids = model.get_text_id(question)
        input_sequence,attention_mask = model.get_sequence(image, q_ids)
    else:
        input_sequence = model.encode_image(image)
        attention_mask = torch.ones(input_sequence.size()[:-1], dtype=torch.int,device=model.device)

    input_sequence = torch.cat((input_sequence, eos_embedding), dim=1)
    attention_mask = torch.cat((attention_mask, eos_attention_mask), dim=1)

    generated_tokens = []

    for _ in range(max_length):
        probabilities = model(input_sequence, attention_mask)
        _, next_token = torch.max(probabilities, dim=-1)
        
        generated_tokens.append(next_token.item())

        next_token_embedding = model.generator.get_input_embeddings()(next_token)
        input_sequence = torch.cat((input_sequence, next_token_embedding.unsqueeze(1)), dim=1)

        new_attention_mask = torch.ones((attention_mask.size(0), 1), dtype=torch.int,device=model.device)
        attention_mask = torch.cat((attention_mask, new_attention_mask), dim=1)

    generated_text = model.generator_tokenizer.decode(generated_tokens)

    return generated_text

def display_captions(dataset:VizWiz,ids: list=[0,1]):
    ids = [id for id in ids if id in dataset.anns]
    
    fig, ax = plt.subplots(len(ids), 1, figsize=(3, 6))

    anns = dataset.loadAnns(ids)
    img_ids = [data['image_id'] for data in anns]
    imgs = dataset.loadImgs(img_ids)

    for i in range(len(ids)):
        file = imgs[i]['file_name']
        img = cv2.cvtColor(cv2.imread(f'Data/Descriptive/train/{file}'), cv2.COLOR_BGR2RGB)
        ax[i].imshow(img)
        ax[i].set_title(anns[i]['caption'], fontsize=8)
        ax[i].axis('off')

    plt.subplots_adjust(wspace=0.5)
    plt.show()

    return None

"""
def get_one_hot_2(indices: torch.Tensor, vocab_size: int,index:int):
    batch_size = len(indices['input_ids'])
    one_hot_batch = torch.zeros(batch_size, vocab_size).to('cuda')
    indices_tensor = indices['input_ids'].squeeze(1)
    for i in range(batch_size):
        ind = indices_tensor[i][index]
        one_hot_batch[i][ind] = 1
    return one_hot_batch
"""

def get_one_hot(indices: torch.Tensor, vocab_size: int):
    input_ids = indices['input_ids'].squeeze(1)
    batch_size, sequence_length = input_ids.size()

    one_hot_batch = torch.zeros(batch_size, sequence_length, vocab_size, device=input_ids.device)

    one_hot_batch.scatter_(2, input_ids.unsqueeze(-1), 1)
    print(one_hot_batch.size())
    return one_hot_batch



def get_eos_embedding(model,batch_size:int):
    eos_id = model.generator_tokenizer.convert_tokens_to_ids('[EOS]')
    eos_id = torch.tensor([eos_id],device=model.device)
    eos_id = eos_id.unsqueeze(0).repeat(batch_size, 1)
    eos_embedding = model.generator.get_input_embeddings()(eos_id)
    eos_attention_mask = torch.zeros((batch_size, 1), dtype=torch.int,device=model.device)
    return eos_embedding,eos_attention_mask