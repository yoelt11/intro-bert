from transformers import BertTokenizer

if __name__=="__main__":

    # -- loading pretrained tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # -- tokenize
    text = "This is a test sentence, let's see how it works!"
    tokens = tokenizer(text)
    print(f'Tokenization outputs: \n {tokens}')

    # -- rebuild text
    new_text = tokenizer.decode(tokens.input_ids)
    print(f'Tokenizer decoding outputs: \n {new_text}')
