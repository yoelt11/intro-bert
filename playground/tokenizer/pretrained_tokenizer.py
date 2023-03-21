from transformers import BertTokenizer
import torch

if __name__=="__main__":

    # -- loading pretrained tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", max_length=30, padding='max_length', truncation=True,  return_type='pt')

    # -- tokenize
    text = "Let us play a game"
    # 101 cls token
    tokens = tokenizer(text)
    print(f'Tokenization outputs: \n {tokens}')
    #input_mask_expanded = torch.tensor(tokens['attention_mask']).unsqueeze(0).unsqueeze(-1).expand([1,16,524])
    #embeddings = torch.randn([1,16,524])
    
    #sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
    #sum_embeddings = torch.sum(embeddings * input_mask_expanded, -1) # it can be along the last dimension or token dimension
    #sum_mask = input_mask_expanded.sum(1)
    #sum_mask = torch.clamp(sum_mask, min=1e-9)

    #output = sum_embeddings / sum_mask
    #print(sum_embeddings.shape)
    #print(sum_mask)
    
    #print(output)
    # token_type_ids: identifies the two types of sequence in the model
    # attention_mask: a mask to identify where the model should pay attention, e.g: on padded inputs, ignore padding tokens

    # -- rebuild text
    new_text = tokenizer.decode(tokens.input_ids)
    print(f'Tokenizer decoding outputs: \n {new_text}')
