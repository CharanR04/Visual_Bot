import torch
from tqdm import tqdm
import torch.nn.functional as F
from src.Tools import get_one_hot,update_sequence_mask,get_eos_embedding

def one_epoch_caption(model,dataloader,optimizer):
    model.train()
    train_loss = 0.0
    model = model.to('cuda')
    print('cuda' if next(model.parameters()).is_cuda else 'cpu')

    for batch_idx, (image,text_ids) in enumerate(tqdm(dataloader, total=len(dataloader), ncols=60)):
        optimizer.zero_grad()
        eos_embedding,eos_attention_mask = get_eos_embedding(model,image.size(0))
        input_sequence = model.encode_image(image)
        attention_mask = torch.ones(input_sequence.size()[:-1], dtype=torch.long).to('cuda')

        input_sequence = torch.cat((input_sequence, eos_embedding), dim=1)
        print(attention_mask.device,eos_attention_mask.device)
        attention_mask = torch.cat((attention_mask, eos_attention_mask), dim=1)
        loss = 0

        for i in range(2):
            probablities = model(input_sequence,attention_mask)
            input_sequence,attention_mask = update_sequence_mask(model,probablities,input_sequence,attention_mask)
            out_encoding = get_one_hot(text_ids,len(model.generator_tokenizer.vocab),i)
            loss += F.cross_entropy(probablities,out_encoding)

        train_loss += loss

        loss.backward()

        optimizer.step()
    print(f'{batch_idx}:{train_loss/len(dataloader)}')

    return train_loss