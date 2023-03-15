import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertTokenizer
from transformers import BertGenerationEncoder

class IntroBert(nn.Module):

    def __init__(self, n_feat=5, max_sentence_len=30):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # -- param
        self.n_feat = n_feat
        self.max_sentence_len = max_sentence_len
        # -- the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # -- the pretrained encoder
        self.encoder = BertGenerationEncoder.from_pretrained("bert-base-uncased")
        self.encoder.to(self.device).train()
        # -- the trainable neural network
        init_size = self.n_feat * self.max_sentence_len * 3
        self.W = torch.rand(init_size, 3, requires_grad=True).to(self.device)
        self.softmax = nn.Softmax(1)
        #self.mlp = nn.Sequential(
          #          nn.Linear(init_size, 3),
                    #nn.GELU(),
                    #nn.Linear(init_size*2, init_size*2),
                    #nn.GELU(),
                    #nn.Linear(init_size*2, init_size*2),
                    #nn.GELU(),
                    #nn.Linear(init_size*2, int(init_size/2)),
                    #nn.GELU(),
                    #nn.Linear(int(init_size/2), int(init_size/4)),
                    #nn.GELU(),
                    #nn.Linear(int(init_size/4), 3),
           #         nn.Softmax(1)
         #       )

        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def tokenize(self, sentence: str, max_sentence_len: int = 30) -> torch.Tensor:
        # -- tokenize input, sets all inputs to equal length according to 'max_seq_len'
        tokens = self.tokenizer(sentence, max_length=max_sentence_len, truncation=True, padding='max_length').input_ids
        return torch.tensor(tokens).to(self.device)

    def get_top_features(self, encoded_tensor:torch.Tensor, n_feat: int=3) -> list[torch.Tensor]:
        """Performs SVD and returns the principal components according to n_feat"""
        u, s, v = torch.svd(encoded_tensor.to('cpu').squeeze(0))
        B_r = u[:, :n_feat] * s[:n_feat] # principal components
        return B_r.to(self.device)
    
    def prepare_inputs(self, sentence_1, sentence_2):
        ''' Gradients should be turned off during training for this module '''
        # -- tokenize inputs
        tokens_1 = self.tokenize(sentence_1)
        tokens_2 = self.tokenize(sentence_2)
        # -- encode inputs
        encoded_1 = self.encoder(tokens_1).last_hidden_state
        encoded_2 = self.encoder(tokens_2).last_hidden_state
        # -- to list
        #encoded_1 = [encoded_1[i] for i in range(encoded_1.shape[0]) ]
        #encoded_2 = [encoded_2[i] for i in range(encoded_2.shape[0]) ]

        #feat_1 = torch.stack(list(map(lambda feature: self.get_top_features(feature, n_feat=self.n_feat), encoded_1)))
        #feat_2 = torch.stack(list(map(lambda feature: self.get_top_features(feature, n_feat=self.n_feat), encoded_2)))
        
        return encoded_1, encoded_2 # feat_1, feat_2
    
    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(output, target) * 10000

    def mean_pool(self, a):
        mean = torch.mean(a, 0, True)
        a[a < mean] = 0
        return a


    def forward(self, sentence_1, sentence_2):
        # -- get inputs
        feat_1, feat_2 = self.prepare_inputs(sentence_1, sentence_2)
        # feat_1 = self.mean_pool(feat_1.flatten(1))
        # feat_2 = self.mean_pool(feat_2.flatten(1))
        #feat_1 = feat_1.flatten(1)
        #feat_2 = feat_2.flatten(1)
        #diff = feat_1.clone() - feat_2.clone()

        # -- concat
        #features = torch.cat((feat_1, feat_2, diff), dim=1)
        output =  feat_1#self.softmax(torch.mm(features,self.W))
        return output

if __name__=='__main__':
    # -- inputs
    sentence_1 = ['this church choir sings to the masses as they sing joyous songs from the book at a church.', 'This church choir sings to the masses as they sing joyous songs from the book at a church.']
    sentence_2 = ['The church is filled with song.', 'A choir singing at a baseball game.']
    # -- target label
    label = torch.tensor([[1.,0.,0.], [0.,0.,1.]]).to('cuda')
    # -- model initialization
    model = IntroBert().to('cuda')
    # -- infer
    output = model(sentence_1, sentence_2)
    #model.loss(output, label).backward()

    # -- analyze
    print(type(output))
    print(output)
    print(output.shape)
