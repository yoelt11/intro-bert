import torch 
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from dataset.textfuse_ds import TextFuseDataset
from transformers import BertGenerationConfig, BertGenerationDecoder, BertGenerationEncoder, EncoderDecoderModel, BertTokenizer
from rouge_score import rouge_scorer


def load_dataset(batch_size, root_path):
    # -- create dataset instance
    dataset = TextFuseDataset(root_path)
    # -- get train-test size
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    # -- use random-split to split data into train and test subsets
    train_dataset , test_dataset = random_split(dataset, [train_size, test_size])
    # -- create dataloaders for train and test datasets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return train_dataloader, test_dataloader

def load_tokenizer():
    # -- loading Tokenizer 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", model_max_length=512)
    return tokenizer

def load_model():

    # -- initializing (pretrained) Encoder
    encoder = BertGenerationEncoder.from_pretrained("bert-base-uncased", 
            bos_token_id=tokenizer.cls_token_id, 
            eos_token_id=tokenizer.sep_token_id) 

    # -- initializing (untrained) decoder BertGeneration config
    config = BertGenerationConfig(bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.sep_token_id)
    config.is_decoder=True
    config.add_cross_attention=True
    
    # -- initializing a model (untrained) decoder from the config
    decoder = BertGenerationDecoder(config)

    # -- combining models into a EncoderDecoderModel (transformers package)
    encoder_decoder = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    encoder_decoder.config.decoder_start_token_id = tokenizer.cls_token_id 
    encoder_decoder.config.pad_token_id = tokenizer.pad_token_id

    return encoder_decoder

def validate():
    # -- set model in evaluation mode
    model.eval()

    # -- metric var
    epoch_precision = 0
    epoch_recall = 0
    epoch_fmeasure = 0

    for i, batch_data in enumerate(train_dataloader, 0):

        # -- get inputs and ouputs
        input_ids = batch_data['input_ids'].squeeze(1)
        labels = batch_data['labels'].squeeze(1)
        labels[labels == -100] = tokenizer.pad_token_id

        # -- run inference
        pred_ids = model.generate(input_ids=input_ids)

        # -- decode strings
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # ?? rouge score does not allow for batched operations?
        def get_scores(pred_str, label_str):
            precision = 0
            recall = 0
            fmeasure = 0

            for i in range(len(pred_str)):
                scores = scorer.score(pred_str[i], label_str[i])
                precision += scores['rouge2'].precision
                recall += scores['rouge2'].recall
                fmeasure += scores['rouge2'].fmeasure
            
            return precision / i, recall / i, fmeasure / i

        precision, recall, fmeasure = get_scores(pred_str, label_str)

        epoch_precision += precision
        epoch_recall += recall
        epoch_fmeasure += fmeasure

        if i % 20 == 0:
            print(f'batched precision: {precision}')
            print(f'batched recall: {recall}')
            print(f'batched fmeasure: {fmeasure}')

            # -- print results
            print(pred_str)
            print(label_str)

    print(f'[Epoch Summary: {epoch + 1}]') 
    print(f'precision: {epoch_precision / BATCH_SIZE:.3f}')
    print(f'recall: {epoch_recall / BATCH_SIZE:.3f}')
    print(f'fmeasure: {epoch_fmeasure / BATCH_SIZE:.3f}')



def train():

    # -- set model training mode
    model.train()

    # -- initialize running_loss for metrics
    running_loss = 0.0
    
    # -- lock encoder's parameters

    for param in model.encoder.parameters(): # only to calculate gradients for decoder
        param.requires_grad = False 

    for i, batch_data in enumerate(train_dataloader, 0):

        # -- get inputs and ouputs
        input_ids = batch_data['input_ids'].squeeze(1)
        labels = batch_data['labels'].squeeze(1)
        
        # -- set gradients to zero
        optimizer.zero_grad()

        # -- run model
        outputs = model(input_ids=input_ids, labels=labels)
        
        # -- calculate losses
        loss = outputs.loss
        loss.backward()
        
        # -- perform optimizer step
        optimizer.step()
        
        # -- in-training metrics
        running_loss += loss.item()
        if i % 20 == 0:
            print(f'[Epoch: {epoch + 1}, iteration: {i + 1:5d}] \nloss: {loss.item() / 100:.3f}')

    print(f'[Epoch: {epoch + 1}] \nloss: {running_loss / BATCH_SIZE:.3f}')



if __name__=="__main__":
    # -- set constants
    SAVE_DIR = '/tmp/model.pt'
    ROOT_PATH = "./dataset/"
    BATCH_SIZE = 2
    LR = 1e-3
    WD = 25e-2
    EPOCHS = 100
    # -- set training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- load dataset
    train_dataloader, test_dataloader = load_dataset(BATCH_SIZE, ROOT_PATH)

    # -- load tokenizer
    tokenizer = load_tokenizer()

    # -- load model
    model = load_model()
    model.to(device)

    # -- load optimzer
    optimizer = optim.AdamW(model.decoder.parameters(),  # optimizing decoder parameters only
            lr=LR,
            weight_decay=WD)

    # -- load metrics
    # rouge-n (n-gram) scoring
    # rouge-l (longest common subsqeuence) scoring
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)

    # --  training loop
    for epoch in range(EPOCHS):
        print(f"----------------- EPOCH: {epoch}---------------------")
        # -- training step
        train()
        # -- validation step
        validate()

        # -- save checkpoint
        #if epoch % 25 == 0:
        #    torch.save(model, "/tmp/intro-bert.pt")



