import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from src.Tools import get_one_hot,update_sequence_mask,get_eos_embedding
import os
import matplotlib.pyplot as plt
import math

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

def train_QA(epoch, model, dataloader, optimizer, version):
    model.set_train()
    print('cuda' if next(model.parameters()).is_cuda else 'cpu')
    model.to(model.device)
    train_loss_history = []

    for e in range(epoch):
        train_loss = 0.0
        for batch_idx, ((image,q_ids),text_ids) in enumerate(tqdm(dataloader, total=len(dataloader), ncols=60)):
            optimizer.zero_grad()
            target_ids = text_ids['input_ids'].squeeze(1).to(model.device)

            gen_emb, gen_mask = get_eos_embedding(model, image['pixel_values'].size(0))

            loss = 0
            loss_function = nn.CrossEntropyLoss()

            for i in range(25):
                probabilities = model(image=image, q_ids=q_ids, gen_emb=gen_emb,gen_mask=gen_mask)
                target = target_ids[:, i]
                
                next_emb = model.generator.get_input_embeddings()(target)
                next_mask = model.get_next_mask(target)

                gen_emb = torch.cat((gen_emb,next_emb.unsqueeze(1)),dim=1)
                gen_mask = torch.cat((gen_mask,next_mask),dim=1)
                
                loss += loss_function(probabilities, target)
                loss.backward(retain_graph = True)
            optimizer.step()

        train_loss += loss.item()

        train_loss_history.append(train_loss / len(dataloader))
        print(f'Epoch {e}: {train_loss / len(dataloader)}')
    
    model.save_model(version)
    return train_loss_history

def train_QA_easy(epoch,model,dataloader,optimizer,version):
    model.set_train()
    print('cuda' if next(model.parameters()).is_cuda else 'cpu')
    model = model.to(model.device)
    train_loss_history = []

    for e in range(epoch):
        train_loss = 0.0
        for batch_idx, ((image,q_ids),text_ids) in enumerate(tqdm(dataloader, total=len(dataloader), ncols=60)):
            optimizer.zero_grad()
            target_ids = text_ids['input_ids'].squeeze(1).to(model.device)
            image = image.to(model.device)
            q_ids = q_ids.to(model.device)

            gen_emb,gen_mask = get_eos_embedding(model,image['pixel_values'].size(0))
            input_sequence,attention_mask = model.get_sequence(image, q_ids)

            input_sequence = torch.cat((input_sequence, gen_emb), dim=1)
            attention_mask = torch.cat((attention_mask, gen_mask), dim=1)
            loss = 0
            loss_function = nn.CrossEntropyLoss()

            for i in range(50):
                probablities = model(input_sequence=input_sequence,attention_mask=attention_mask)

                target = target_ids[:, i]
                
                next_emb = model.generator.get_input_embeddings()(target)
                next_mask = model.get_next_mask(target)

                input_sequence = torch.cat((gen_emb,next_emb.unsqueeze(1)),dim=1)
                attention_mask = torch.cat((gen_mask,next_mask),dim=1)
                
                loss += loss_function(probablities, target)
            

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss_history.append(train_loss/len(dataloader))
        print(f'{epoch}:{train_loss/len(dataloader)}')
    model.save_model(version)
    return train_loss_history

def _train_caption(epoch, model, dataloader, optimizer, version):
    torch.autograd.set_detect_anomaly(True)
    model.set_train()
    print('cuda' if next(model.parameters()).is_cuda else 'cpu')
    model.to(model.device)
    train_loss_history = []
    grad_norms = []
    raw_grad_norms= []

    for e in range(epoch):
        train_loss = 0.0
        for batch_idx, (image, text_ids) in enumerate(tqdm(dataloader, total=len(dataloader), ncols=60)):
            target_ids = text_ids['input_ids'].squeeze(1).to(model.device)

            gen_emb, gen_mask = get_eos_embedding(model, image['pixel_values'].size(0))

            loss_function = nn.CrossEntropyLoss()

            loss = 0.0

            optimizer.zero_grad()
            
            for i in range(25):
                logits = model(image=image,gen_emb=gen_emb,gen_mask=gen_mask,q_ids=None)
                target = target_ids[:, i]

                next_emb = model.generator.get_input_embeddings()(target)
                next_mask = model.get_next_mask(target)
        
                gen_emb = torch.cat((gen_emb,next_emb.unsqueeze(1)),dim=1)
                gen_mask = torch.cat((gen_mask,next_mask),dim=1)
                loss += loss_function(logits[:,-1,:], target)
            loss.backward()
            clipped_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if clipped_grad_norm is not None:
                raw_grad_norms.append(clipped_grad_norm.item())

            # Calculate and append the raw gradient norm only if gradients are not None
            raw_grad_norm = torch.norm(torch.stack([p.grad.detach().norm() for p in model.parameters() if p.grad is not None]), 2)
            if raw_grad_norm.numel() > 0:  # Check if there are gradients to compute the norm
                grad_norms.append(raw_grad_norm.item())

            # Check if clipping has occurred
            if clipped_grad_norm is not None and raw_grad_norm.numel() > 0:
                if clipped_grad_norm < raw_grad_norm:
                    print("Gradient clipping occurred!")
            optimizer.step()

   
            if batch_idx%100 == 1:
                print(train_loss/batch_idx)
                plt.figure(figsize=(6, 3))
                #plt.plot(raw_grad_norms, label='Raw Gradient Norms')
                plt.plot(grad_norms, label='Clipped Gradient Norms')
                plt.title('Gradient Norms During Training')
                plt.xlabel('Iterations')
                plt.ylabel('Norm')
                plt.legend()
                plt.show()
                grad_norms = []
                raw_grad_norms= []

            train_loss += loss.item()
        if e < 12:
            for name,param in model.generator.named_parameters():
                if name == model.params[e]:
                    param.requires_grad_(True)

        train_loss_history.append(train_loss / len(dataloader))
        print(f'Epoch {e}: {train_loss / len(dataloader)}')
    
    model.save_model(version)
    return train_loss_history

def train_caption(epoch, model, dataloader, optimizer, version):
    torch.autograd.set_detect_anomaly(False)
    model.set_train()
    print('cuda' if next(model.parameters()).is_cuda else 'cpu')
    model.to(model.device)
    train_loss_history = []
    grad_norms = []
    raw_grad_norms= []
    scaler = torch.cuda.amp.GradScaler()
    for e in range(epoch):
        train_loss = 0.0
        for batch_idx, (image, text_ids) in enumerate(tqdm(dataloader, total=len(dataloader), ncols=60)):
            target_ids = text_ids['input_ids'].squeeze(1).to(model.device)
            loss_function = nn.CrossEntropyLoss()
            image = image.to(model.device)
            embedding = model.encode_image(image)
            mask = torch.ones(embedding.size()[:-1], dtype=torch.long, device=model.device)
            loss = 0.0
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():  # Enable mixed precision
                for i in range(25):
                    logits = model(embedding, mask)
                    target = target_ids[:, i]

                    next_emb = model.generator.get_input_embeddings()(target)
                    next_mask = model.get_next_mask(target)

                    embedding = torch.cat((embedding, next_emb.unsqueeze(1)), dim=1)
                    mask = torch.cat((mask, next_mask), dim=1)
                    loss += loss_function(logits[:, -1, :], target)
            
            scaler.scale(loss).backward()
            clipped_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            """
            if clipped_grad_norm is not None:
                raw_grad_norms.append(clipped_grad_norm.item())

            raw_grad_norm = torch.norm(torch.stack([p.grad.detach().norm() for p in model.parameters() if p.grad is not None]), 2)
            if raw_grad_norm.numel() > 0:  # Check if there are gradients to compute the norm
                grad_norms.append(raw_grad_norm.item())

            # Check if clipping has occurred
            if clipped_grad_norm is not None and raw_grad_norm.numel() > 0:
                if clipped_grad_norm < raw_grad_norm:
                    print("Gradient clipping occurred!")
            

            if batch_idx % 100 == 1:
                    print(train_loss / batch_idx)
                    plt.figure(figsize=(6, 3))
                    plt.plot(grad_norms, label='Clipped Gradient Norms')
                    plt.title('Gradient Norms During Training')
                    plt.xlabel('Iterations')
                    plt.ylabel('Norm')
                    plt.legend()
                    plt.show()
                    grad_norms = []
                    raw_grad_norms = []
            """
            train_loss += loss.item()
        train_loss_history.append(train_loss / len(dataloader))
        print(f'Epoch {e}: {train_loss / len(dataloader)}')
        model.save_model(version)
    return train_loss_history

def train_caption_easy(epoch, model, dataloader, optimizer, version):
    model.train()
    print('cuda' if next(model.parameters()).is_cuda else 'cpu')
    model.to(model.device)
    train_loss_history = []

    for e in range(epoch):
        train_loss = 0.0
        for batch_idx, (image, text_ids) in enumerate(tqdm(dataloader, total=len(dataloader), ncols=60)):
            optimizer.zero_grad()
            target_ids = text_ids['input_ids'].squeeze(1).to(model.device)
            image = image.to(model.device)

            gen_emb, gen_mask = get_eos_embedding(model, image['pixel_values'].size(0))

            input_sequence = model.encode_image(image).requires_grad_(True)
            attention_mask = torch.ones(input_sequence.size()[:-1], dtype=torch.long, device=model.device)

            input_sequence = torch.cat((input_sequence, gen_emb), dim=1)
            attention_mask = torch.cat((attention_mask, gen_mask), dim=1)
            loss = 0
            loss_function = nn.CrossEntropyLoss()

            for i in range(25):
                probabilities = model(input_sequence=input_sequence,attention_mask=attention_mask)
                target = target_ids[:, i]
                
                next_emb = model.generator.get_input_embeddings()(target)
                next_mask = model.get_next_mask(target)

                input_sequence = torch.cat((gen_emb,next_emb.unsqueeze(1)),dim=1)
                attention_mask = torch.cat((gen_mask,next_mask),dim=1)
                
                loss_t = loss_function(probabilities, target)+10
                loss += loss_t

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_history.append(train_loss / len(dataloader))
        print(f'Epoch {e}: {train_loss / len(dataloader)}')
    
    model.save_model(version)
    return train_loss_history

def train_vit(epoch, model, dataloader, optimizer,version):
    model.train()
    print('cuda' if next(model.parameters()).is_cuda else 'cpu')
    model.to('cuda')
    train_loss_history = []
    loss_fn = nn.CrossEntropyLoss()
    
    for e in range(epoch):
        train_loss = 0.0
        for batch_idx, (image, labels) in enumerate(tqdm(dataloader, total=len(dataloader), ncols=60)):
            optimizer.zero_grad()
            outs = model(image)
            loss = loss_fn(outs,labels)
            loss.backward()
            optimizer.step()
            train_loss+=loss
            if batch_idx%1000 == 0:
                print(f'loss=> {e}:{train_loss/len(dataloader)}')
        print(f'loss=> {e}:{train_loss/len(dataloader)}')
        train_loss_history.append(train_loss/len(dataloader))


    version= 'v_'+str(version)
    if not os.path.exists(os.path.join('Model','VIT', version)):
        os.makedirs(os.path.join('Model','VIT', version))
    model.model.save_pretrained(os.path.join('Model','VIT', version,'VIT'))
    return train_loss_history