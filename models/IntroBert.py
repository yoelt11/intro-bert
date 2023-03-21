import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config, MT5Config, T5EncoderModel

class IntroBert(nn.Module):

    def __init__(self, n_feat=5, max_sentence_len=30, pooling_stg='mean_pool'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # -- param
        self.n_feat = n_feat
        self.max_sentence_len = max_sentence_len
        # -- the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        # -- the pretrained encoder
        T5EncoderModel.__key_to_ignore_on_load_unexpected = ["decoder.*"]
        config = AutoConfig.from_pretrained("t5-base")
        self.encoder = T5EncoderModel.from_pretrained("t5-base")
        self.encoder.to(self.device).train()
        # -- pooling strategy
        match pooling_stg:
            case "mean_pool":
                self.pooling = self.mean_pool
            case "cls_pool":
                self.pooling = self.cls_pool
        # -- the trainable neural network
        init_size = 1537
        self.W = torch.rand(init_size, 3, requires_grad=True).to(self.device)
        self.sigma = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def tokenize(self, sentence: str, max_sentence_len: int = 30) -> torch.Tensor:
        # -- tokenize input, sets all inputs to equal length according to 'max_seq_len'
        tokens = self.tokenizer(sentence, max_length=max_sentence_len, truncation=True, padding='max_length').input_ids
        return torch.tensor(tokens).to(self.device)

    def encode(self, sentence: str) -> torch.Tensor:
        # -- In inference only the trained encoder, and sentences compared with cosine_similarity
        tokens = self.tokenize(sentence).unsqueeze(0)
        encoded = self.encoder(tokens).last_hidden_state.flatten()

        return encoded

    def prepare_inputs(self, sentence_1: str, sentence_2: str) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' Gradients should be turned off during training for this module '''
        # -- tokenize inputs
        tokens_1 = self.tokenize(sentence_1)
        tokens_2 = self.tokenize(sentence_2)
        # -- encode inputs
        encoded_1 = self.encoder(tokens_1).last_hidden_state
        encoded_2 = self.encoder(tokens_2).last_hidden_state

        return encoded_1, encoded_2 # feat_1, feat_2
    
    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(output, target) * 10000

    def mean_pool(self, a):
        mean = torch.mean(a, dim=1)
        return torch.clamp(mean, min=1e-9)

    def cls_pool(self, a):
        return a[:,0,:]

    def forward(self, sentence_1, sentence_2):
        # -- get inputs
        feat_1, feat_2 = self.prepare_inputs(sentence_1, sentence_2) 

        feat_1 = self.pooling(feat_1)
        feat_2 = self.pooling(feat_2)

        # diff = self.sigma(feat_1 - feat_2)
        diff = (feat_1 - feat_2).pow(2).sum(1).sqrt().unsqueeze(-1)

        # -- concat
        features = torch.cat((feat_1, feat_2, diff), dim=1)
        output =  self.softmax(torch.mm(features,self.W))
        return output

if __name__=='__main__':
    # -- inputs
    sentence_1 = ['this church choir sings to the masses as they sing joyous songs from the book at a church.', 'This church choir sings to the masses as they sing joyous songs from the book at a church.']
    sentence_2 = ['The church is filled with song.', 'A choir singing at a baseball game.']
    # -- target label
    label = torch.tensor([[1.,0.,0.], [0.,0.,1.]]).to('cuda')
    # -- model initialization
    model = IntroBert(pooling_stg="cls_pool").to('cuda')
    # -- infer
    output = model(sentence_1, sentence_2)
    model.loss(output, label).backward()

    # -- analyze
    print(type(output))
    print(output)
    print(output.shape)
