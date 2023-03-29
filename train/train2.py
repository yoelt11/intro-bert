import torch 
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from dataset import SNLI
import sys
sys.path.append('../models')
from torch.optim.lr_scheduler import LinearLR
from SemSim import SemSim
from matplotlib import pyplot as plt

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
    epoch_loss = 0.0 
    for i, batch_data in enumerate(train_dataloader, 0):
        running_loss = model.fit(batch_data, optimizer, epoch)

        epoch_loss += running_loss # / BATCH_SIZE

        if i % 20 == 0:
            print(f"[Train][Epoch: {epoch}][Batch: {i}] Loss: {epoch_loss}")
    return epoch_loss / i

def validate():
    # -- set model in evaluation mode
    model.eval()

    # -- metric var
    epoch_score = 0
    
    for i, batch_data in enumerate(test_dataloader, 0):
       score = model.test(batch_data, epoch)
       epoch_score += score / BATCH_SIZE
       if i % 20 == 0:
            print(f"[Test][Epoch: {epoch}][Batch: {i}] Score: {epoch_score}")
        
    return score / i

def plot_metrics(train_metrics, test_metrics):
    with torch.no_grad():
        fig = plt.figure(figsize=(8, 8))
        fig.tight_layout()
        # -- plot only the ith value
        # -- emb 1
        ax1 = fig.add_subplot(211)
        ax1.title.set_text("train_metrics: loss")
        ax1.plot(train_metrics)
        # -- emb 2
        ax2 = fig.add_subplot(212)
        ax2.title.set_text("test_metrics: score")
        ax2.plot(test_metrics)
        # -- show or save image
        plt.savefig(f'../../metrics_epoch_{str(epoch)}.png')
        plt.close()
        
            
if __name__=="__main__":
    # -- set constants
    SAVE_DIR = '/tmp/model.pt'
    ROOT_PATH = '/home/edgar/Documents/in-work/datasets/snli_1.0/snli_1.0/'
    BATCH_SIZE = 24
    LR = 2e-5
    WD = 25e-2
    EPOCHS = 100
    # -- set training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- load dataset
    train_dataloader, test_dataloader = load_dataset(BATCH_SIZE, ROOT_PATH)

    # -- load model
    model = SemSim()
    model.to(device)

    # -- load optimzer
    optimizer = optim.AdamW(model.parameters(),  # optimizing decoder parameters only
            lr=LR,
            weight_decay=WD)
    scheduler = LinearLR(optimizer, start_factor = 0.5, total_iters=1000)

    # -- metrics
    train_metrics = []
    test_metrics = []

    # --  training loop
    for epoch in range(EPOCHS):
        print(f"----------------- EPOCH: {epoch}---------------------")
        # -- training step
        metric = train()
        train_metrics.append(metric)
        # -- validation step
        metric = validate()
        test_metrics.append(metric)

        # -- plot progress
        plot_metrics(train_metrics, test_metrics)
        # -- save checkpoint
        if epoch % 25 == 0:
            torch.save(model, "../../intro-bert.pt")

