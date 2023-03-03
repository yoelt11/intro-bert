from transformers import BertTokenizer, BertModel, BertForPreTraining
import torch

if __name__== "__main__":
    
    # -- load pretrained tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # -- load pretrained model
    model = BertForPreTraining.from_pretrained('bert-base-uncased') # model = BertModel.from_pretrained('bert-base-uncased')

    # -- dummy input 
    text = "This is a test sentence, let's see how it [MASK]!"
    tokens = tokenizer(text, return_tensors="pt")

    # -- run model
    outputs = model(**tokens)

    # get items with higher score and decode
    prediction = outputs.prediction_logits.squeeze(0).argmax(dim=1)
    print(tokenizer.decode(prediction))
    

    #print(outputs.prediction_logits.shape)
    #print(outputs.seq_relationship_logits)
    #last_hidden_state = outputs 
    #print("outputs: \n",  tokenizer.decode(last_hidden_state))
    #print(outputs.prediction_logits.squeeze(0).argmax(dim=1).shape)
