import torch
from torch.utils.data import Dataset, DataLoader
import json

class SNLI(Dataset):

    def __init__(self, path):
        
        with open(path, 'r') as file:
            self.dataset = [json.loads(jline) for jline in list(file)]
        
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''if golden_label=='-' item should be skipped'''
        return self.dataset[idx]['sentence1'], self.dataset[idx]['sentence2'], self.label_encode(self.dataset[idx]['gold_label'])

    def label_encode(self, label):
        output = torch.zeros(3)
        if label == 'contradiction':
            output[0] = 1.0
        elif label == 'neutral':
            output[1] = 1.0
        elif label == 'entailment':
            output[2] = 1.0

        return output # if unusable then should return [0,0,0]


if __name__=='__main__':
    dataset = SNLI('../playground/dataset-analysis/snli_1.0_train.jsonl')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(next(iter(dataloader)))
