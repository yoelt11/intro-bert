from transformers import BertTokenizer
import torch
from torch.nn import functional as F

if __name__=="__main__":

    # -- loading pretrained tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # -- tokenize sentence A
    text = "is verything fine?"
    tokens_a = tokenizer(text, max_length=20).input_ids
    cl = len(tokens_a)
    ps = 10 - cl
    tokens_a = F.pad(torch.tensor(tokens_a), [-1,ps],"constant",0)
    print(f'Tokenization outputs: \n {tokens_a}')

    # -- tokenize sentence B
    text = "how are you?"
    tokens_b = tokenizer(text, max_length=20).input_ids
    cl = len(tokens_b)
    ps = 10 - cl
    tokens_b = F.pad(torch.tensor(tokens_b), [-1,ps],"constant",0)
    print(f'Tokenization outputs: \n {tokens_b}')

    # -- tokenize sentence C
    text = "are you ok?"
    tokens_c = tokenizer(text, max_length=20).input_ids
    cl = len(tokens_c)
    ps = 10 - cl
    tokens_c = F.pad(torch.tensor(tokens_c), [-1,ps],"constant",0)
    print(f'Tokenization outputs: \n {tokens_c}')

    # -- rebuild text
    new_text = tokenizer.decode(tokens_c)
    print(f'Tokenizer decoding outputs: \n {new_text}')

    # -- diff a-b, b-c
    print(tokens_b-tokens_a)
    print(tokens_b-tokens_c)
