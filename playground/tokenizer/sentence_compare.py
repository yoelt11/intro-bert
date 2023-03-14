from transformers import BertTokenizer

if __name__=="__main__":

    # -- loading pretrained tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # -- tokenize sentence A
    text = "is verything fine?"
    tokens_a = tokenizer(text, max_length=20)
    print(f'Tokenization outputs: \n {tokens_a}')

    # -- tokenize sentence B
    text = "how are you?"
    tokens_b = tokenizer(text, max_length=20)
    print(f'Tokenization outputs: \n {tokens_b}')

    # -- tokenize sentence C
    text = "are you ok?"
    tokens_c = tokenizer(text, max_length=20, padding_side='right')
    print(f'Tokenization outputs: \n {tokens_c}')

    # -- rebuild text
    new_text = tokenizer.decode(tokens_c)
    print(f'Tokenizer decoding outputs: \n {new_text}')
