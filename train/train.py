import torch 
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from dataset import SNLI
import sys
sys.path.append('../models')
from torch.optim.lr_scheduler import LinearLR
from IntroBert import IntroBert

def load_dataset(batch_size, root_path):
    # -- create dataset instance
    train_dataset = SNLI(root_path + 'snli_1.0_train.jsonl')
    test_dataset = SNLI(root_path + 'snli_1.0_test.jsonl')
    # -- create dataset loader 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return train_dataloader, test_dataloader

def train():
    # -- set model training mode
    model.train()

    # -- initialize running_loss for metrics
    running_loss = 0.0
    
    # -- lock encoder's parameters

    #for param in model.encoder.parameters(): # only to calculate gradients for decoder
    #    param.requires_grad = False 

    for i, batch_data in enumerate(train_dataloader, 0):

        # -- get inputs and ouputs
        sentence_1 = batch_data[0]
        sentence_2 = batch_data[1]
        targets = batch_data[2].to(device)
        
        # -- set gradients to zero
        optimizer.zero_grad()

        # -- run model
        outputs = model(sentence_1, sentence_2)
        
        # -- calculate losses
        loss = model.loss(outputs, targets )
        loss.backward()
        
        # -- perform optimizer step
        optimizer.step()
        
        # -- in-training metrics
        running_loss += loss.item()
        scheduler.step()
        if i % 20 == 0:
            print(f'[Epoch: {epoch + 1}, iteration: {i + 1:5d}] \nloss: {loss.item() / 100:.3f}')
            acc = (targets.detach().cpu().numpy().argmax(1) == outputs.detach().cpu().numpy().argmax(1)).sum() /  BATCH_SIZE
            print(f'batched precision: {acc}')

    print(f'[Epoch: {epoch + 1}] \nloss: {running_loss / BATCH_SIZE:.3f}')

def validate():
    # -- set model in evaluation mode
    model.eval()

    # -- metric var
    epoch_precision = 0
    with torch.no_grad():
        for i, batch_data in enumerate(test_dataloader, 0):

            # -- get inputs and ouputs
            sentence_1 = batch_data[0]
            sentence_2 = batch_data[1]
            targets = batch_data[2].to(device)

            # -- run model
            outputs = model(sentence_1, sentence_2)
            
            acc = (targets.argmax(1) == outputs.argmax(1)).sum() /  BATCH_SIZE

            epoch_precision += acc
            if i % 20 == 0:
                print(f'batched precision: {acc}')
                print(f'sentence a: {sentence_1[0]} \nsentence b: {sentence_2[0]} \nprediction: {outputs[0].argmax()}')

    print(f'[Epoch Summary: {epoch + 1}]') 
    print(f'precision: {epoch_precision / i:.3f}')

if __name__=="__main__":
    # -- set constants
    SAVE_DIR = '/tmp/model.pt'
    ROOT_PATH = '../../../../in-work/datasets/snli_1.0/'
    BATCH_SIZE = 24
    LR = 2e-5
    WD = 25e-2
    EPOCHS = 100
    # -- set training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- load dataset
    train_dataloader, test_dataloader = load_dataset(BATCH_SIZE, ROOT_PATH)

    # -- load model
    model = IntroBert()
    model.to(device)

    # -- load optimzer
    optimizer = optim.AdamW(model.parameters(),  # optimizing decoder parameters only
            lr=LR,
            weight_decay=WD)
    scheduler = LinearLR(optimizer, start_factor = 0.5, total_iters=1000)
    # --  training loop
    for epoch in range(EPOCHS):
        print(f"----------------- EPOCH: {epoch}---------------------")
       # -- training step
        train()
       # -- validation step
        validate()
        # -- save checkpoint
        if epoch % 25 == 0:
            torch.save(model, "../../intro-bert.pt")

