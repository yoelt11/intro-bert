import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config, MT5Config, T5EncoderModel
from typing import Callable

class SemSim(nn.Module):
    
    def __init__(self, max_sentence_len=30, pooling_stg='mean_pool', device=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_sentence_len = max_sentence_len
        # -- initialize components
        self.tokenize = AutoTokenizer.from_pretrained("t5-base", model_max_length=768)
        config = AutoConfig.from_pretrained("t5-base")
        T5EncoderModel._key_to_ignore_on_load_unexpected = ["decoder.*"]
        self.encoder = T5EncoderModel.from_pretrained("t5-base",config=config)#.encoder
        # -- pooling
        self.pool = self.pooling(pooling_stg)
        # -- last layers
        self.linear = nn.Linear(768*3,3)
        self.softmax = nn.Softmax(1)
        self.loss_fn = nn.CrossEntropyLoss()
        # -- for inference
        self.cos_sim = nn.CosineSimilarity()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._target_device = torch.device(device)

    def fit(self, batch_data, optimizer, loss_fn):
        # -- prepare inputs
        sentence_1, sentence_2, targets = batch_data
        targets = targets.to(self.device)
        # -- set gradients to zero
        optimizer.zero_grad()
        # -- run forward pass
        outputs = self.forward(sentence_1, sentence_2)
        # -- calculate loss
        loss = self.loss_fn(outputs, targets)
        # -- backpropagate
        loss.backward()
        # -- update weights
        optimizer.step()

        # -- return running loss
        running_loss = loss.detach().item()

        return running_loss

    def test(self, batch_data, epoch):
        with torch.no_grad():
            # -- get inputs
            sentence_1, sentence_2, targets = batch_data
            targets = targets.to(self.device)
            # -- run model
            outputs = self.infer(sentence_1, sentence_2, epoch=epoch)
            # -- get score 
            matching = outputs.clone()
            matching[matching >= 0.5] = True
            matching[matching < 0.5] = False
            matching =[bool(item) for item in matching.tolist()]

            target_match = [True if item[0] == 0 else False for item in targets ]
            score = torch.tensor([True if matching[i] == target_match[i] else False for i in range(targets.size(0))]).sum() / targets.size(0)
            return score


    def tokenize_encode(self, sentence: str) -> torch.Tensor:
        '''Tokenize and produce embeddings'''
        # -- tokenize input
        tokens = self.tokenize(sentence,return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_sentence_len).input_ids
        # -- encode input
        encode = self.encoder(tokens).last_hidden_state
        
        return encode

    def plot_emb(self, emb_1, emb_2, sent_1, sent_2, epoch):
        with torch.no_grad():
            fig = plt.figure(figsize=(8, 8))
            fig.tight_layout()
            # -- plot only the ith value
            i = torch.randint(low=0, high=emb_1.size(0), size=(1,))
            # -- emb 1
            e_1 = torch.reshape(emb_1[i], (128, -1))
            ax1 = fig.add_subplot(211)
            ax1.title.set_text(sent_1[i])
            ax1.imshow(e_1)
            # -- emb 2
            e_2 = torch.reshape(emb_2[i], (128, -1))
            ax2 = fig.add_subplot(212)
            ax2.title.set_text(sent_2[i])
            ax2.imshow(e_2)
            # -- show or save image
            plt.savefig(f'../../debug_epoch_{str(epoch)}.png')
            plt.close()
    
    def pooling(self, pooling_stg: str) -> Callable[[torch.Tensor], torch.Tensor]:
        match pooling_stg:
            case 'mean_pool':
                def mean_pool(x):
                    return torch.clamp(torch.mean(x, dim=1), 1e-9)
                return mean_pool
            case 'cls_pool':
                def cls_pool(x):
                    return x[:,0,:]
                return cls_pool

    def forward(self, sentence_1: str, sentence_2: str) -> torch.Tensor:
        '''Function ran during training, returns [0,1,0] a 3 dim tensor representing the class.
           First one being not matching, entailment and match'''
        
        # -- tokenize and encode
        emb_1 = self.tokenize_encode(sentence_1)
        emb_2 = self.tokenize_encode(sentence_2)
        # -- concatanate [pool(emb_1), pool(emb_2), diff]
        concat_out = torch.cat([self.pool(emb_1), self.pool(emb_2), self.pool(emb_1) - self.pool(emb_2)], dim=-1)

        return self.softmax(self.linear(concat_out))
    
    def infer(self, sentence_1: str, sentence_2:str, epoch=None) -> torch.Tensor:
        with torch.no_grad():
            # -- tokenize and encode
            emb_1 = self.tokenize_encode(sentence_1)
            emb_2 = self.tokenize_encode(sentence_2)
            self.plot_emb(emb_1, emb_2, sentence_1, sentence_2, epoch)
            return self.cos_sim(emb_1.flatten(1), emb_2.flatten(1))

# -- Testing Zone -- #
if __name__ == '__main__':
    # -- inputs
    sentence_1 = ['this church choir sings to the masses as they sing joyous songs from the book at a church.', 'This church choir sings to the masses as they sing joyous songs from the book at a church.']
    sentence_2 = ['The church is filled with song.', 'A choir singing at a baseball game.']
    # -- target label
    label = torch.tensor([[1.,0.,0.], [0.,0.,1.]])
    # -- model initialization
    model = SemSim(pooling_stg="mean_pool")
    # -- infer
    output = model.test((sentence_1, sentence_2, label),1)
    print(output)
    #print(output.shape)
