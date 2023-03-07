import torch 
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from dataset.textfuse_ds import TextFuseDataset
from transformers import BertGenerationConfig, BertGenerationDecoder, BertGenerationEncoder, EncoderDecoderModel, BertTokenizer

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
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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
    pass

def train():
    pass

if __name__=="__main__":
    # -- set constants
    SAVE_DIR = '/tmp/model.pt'
    ROOT_PATH = "./dataset/"
    BATCH_SIZE = 5
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

    # -- load optimzer
    optimizer = optim.AdamW(model.parameters(), 
            lr=LR,
            weight_decay=WD)

    for epoch in range(EPOCHS):
        # -- training step
        train()
        # -- validation step
        validate()

        # -- save checkpoint
        if epoch % 25 == 0:
            torch.save()



