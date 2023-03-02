from transformers import BertTokenizer, BertForPreTraining
import torch

if __name__== "__main__":

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForPreTraining.from_pretrained('bert-base-uncased')
    
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    print("inputs: ",  inputs)
    outputs = model(**inputs)
    
    prediction_scores, seq_relationship_scores = outputs[:2]
    
    print("outputs: \n",  outputs)
