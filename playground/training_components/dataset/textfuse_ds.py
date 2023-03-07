import yaml
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


class TextFuseDataset(Dataset):

    def __init__(self, data_root):
        # -- open dataset
        with open(data_root + "dataset_example.yaml", "r") as stream:
            samples = yaml.safe_load(stream)
        self.samples = samples
        self.length = len(self.samples)
        self.encoder_max_length = 512
        self.decoder_max_length = 128
        

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # -- Tokenize Input
        inputs = tokenizer(self.samples[idx]['long_text'] , padding='max_length', truncation=True, max_length=self.encoder_max_length)
        outputs = tokenizer(self.samples[idx]['short_text'], padding='max_length', truncation=True, max_length=self.decoder_max_length)
        # -- Create input and label dictionaries
        batch = {}
        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
   
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()
        
        # -- Replace [PAD] with with [UNK] as per BERT, We need to ingore pad tokens
        batch["labels"] = [-100 if token == tokenizer.pad_token_id else token for token in batch["labels"]]

        return batch 

if __name__=='__main__':
    # -- initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # -- initialize dataset
    dataset = TextFuseDataset('./')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # dataset test
    print("---------------- Encoded Input -----------")
    print(torch.tensor(next(iter(dataloader))['input_ids']))
    print(torch.tensor(next(iter(dataloader))['labels']))
    
    print("---------------- Decoded Input -----------")
    print(tokenizer.decode(torch.tensor(next(iter(dataloader))['input_ids'])))
    print(tokenizer.decode(torch.tensor(next(iter(dataloader))['labels'])))

