import torch
from tqdm import tqdm
import torch.nn.functional as F
from src.Tools import get_one_hot,update_sequence_mask,get_eos_embedding

def one_epoch_caption(model,dataloader,optimizer):
    model.set_train()
    train_loss = 0.0
    model = model.to('cuda')
    print('cuda' if next(model.parameters()).is_cuda else 'cpu')

    for batch_idx, (image,text_ids) in enumerate(tqdm(dataloader, total=len(dataloader), ncols=60)):
        optimizer.zero_grad()
        target_ids = text_ids['input_ids'].squeeze(1)

        target_ids = target_ids.to(model.device)

        eos_embedding,eos_attention_mask = get_eos_embedding(model,image.size(0))
        input_sequence = model.encode_image(image)
        attention_mask = torch.ones(input_sequence.size()[:-1], dtype=torch.long,device=model.device)

        input_sequence = torch.cat((input_sequence, eos_embedding), dim=1)
        attention_mask = torch.cat((attention_mask, eos_attention_mask), dim=1)
        loss = 0

        for i in range(50):
            probabilities = model(input_sequence,attention_mask)
            input_sequence,attention_mask = update_sequence_mask(model,probabilities,input_sequence,attention_mask)

            target = target_ids[:, i]

            loss += F.cross_entropy(probabilities, target)

        train_loss += loss

        loss.backward()

        optimizer.step()
    print(f'{batch_idx}:{train_loss/len(dataloader)}')

    return train_loss

def one_epoch_QA(model,dataloader,optimizer):
    model.set_train()
    train_loss = 0.0
    model = model.to('cuda')
    print('cuda' if next(model.parameters()).is_cuda else 'cpu')

    for batch_idx, ((image,q_ids),text_ids) in enumerate(tqdm(dataloader, total=len(dataloader), ncols=60)):
        optimizer.zero_grad()
        target_ids = text_ids['input_ids'].squeeze(1)
        eos_embedding,eos_attention_mask = get_eos_embedding(model,image.size(0))
        input_sequence,attention_mask = model.get_sequence(image, q_ids)

        input_sequence = torch.cat((input_sequence, eos_embedding), dim=1)
        attention_mask = torch.cat((attention_mask, eos_attention_mask), dim=1)
        loss = 0

        for i in range(5):
            probablities = model(input_sequence,attention_mask)
            input_sequence,attention_mask = update_sequence_mask(model,probablities,input_sequence,attention_mask)

            target = target_ids[:, i]

            loss += F.cross_entropy(probablities,target)

        train_loss += loss

        loss.backward()

        optimizer.step()
    print(f'{batch_idx}:{train_loss/len(dataloader)}')

    return train_loss

def train_QA(epoch,model,dataloader,optimizer,version):
    model.set_train()
    model = model.to('cuda')
    print('cuda' if next(model.parameters()).is_cuda else 'cpu')
    train_loss_history = []

    for e in range(epoch):
        train_loss = 0.0
        for batch_idx, ((image,q_ids),text_ids) in enumerate(tqdm(dataloader, total=len(dataloader), ncols=60)):
            optimizer.zero_grad()
            target_ids = text_ids['input_ids'].squeeze(1)
            eos_embedding,eos_attention_mask = get_eos_embedding(model,image.size(0))
            input_sequence,attention_mask = model.get_sequence(image, q_ids)

            input_sequence = torch.cat((input_sequence, eos_embedding), dim=1)
            attention_mask = torch.cat((attention_mask, eos_attention_mask), dim=1)
            loss = 0

            for i in range(5):
                probablities = model(input_sequence,attention_mask)
                input_sequence,attention_mask = update_sequence_mask(model,probablities,input_sequence,attention_mask)

                target = target_ids[:, i].to(model.device)
                loss += F.cross_entropy(probablities,target)
            

            train_loss += loss

            loss.backward()

            optimizer.step()

        train_loss_history.append(train_loss/len(dataloader))
        print(f'{epoch}:{train_loss/len(dataloader)}')
    model.save_model(version)
    return train_loss_history

def train_caption(epoch,model,dataloader,optimizer,version):
    model.set_train()
    model = model.to('cuda')
    print('cuda' if next(model.parameters()).is_cuda else 'cpu')
    train_loss_history = []
    
    for e in range(epoch):
        train_loss = 0.0
        for batch_idx, (image,text_ids) in enumerate(tqdm(dataloader, total=len(dataloader), ncols=60)):
            optimizer.zero_grad()
            target_ids = text_ids['input_ids'].squeeze(1)

            target_ids = target_ids.to(model.device)

            eos_embedding,eos_attention_mask = get_eos_embedding(model,image.size(0))
            input_sequence = model.encode_image(image)
            attention_mask = torch.ones(input_sequence.size()[:-1], dtype=torch.long,device=model.device)

            input_sequence = torch.cat((input_sequence, eos_embedding), dim=1)
            attention_mask = torch.cat((attention_mask, eos_attention_mask), dim=1)
            loss = 0

            for i in range(50):
                probabilities = model(input_sequence,attention_mask)
                input_sequence,attention_mask = update_sequence_mask(model,probabilities,input_sequence,attention_mask)

                target = target_ids[:, i]

                loss += F.cross_entropy(probabilities, target)

            train_loss += loss

            loss.backward()

            optimizer.step()

        train_loss_history.append(train_loss/len(dataloader))
        print(f'{e}:{train_loss/len(dataloader)}')
    model.save_model(version)
    return train_loss_history