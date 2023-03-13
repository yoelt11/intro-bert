from transformers import EncoderDecoderModel, BertConfig, EncoderDecoderConfig, BertGenerationEncoder, BertTokenizer

# -- The idea is to possibly initialize a pretrained bert encoder, and to train a decoder for the specified case. A starting would be to use an untrained bert decoder, train, test, and compare to different similar decoders architectures.

# -- In this script we simply care about loading the pretrained bert encoder.

if __name__=='__main__':

    # -- load pretrained tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # -- dummy input
    text = "This is a test sentence, let's see how it works"
    tokens = tokenizer(text, return_tensors="pt").input_ids

    # -- initialize encoder
    encoder = BertGenerationEncoder.from_pretrained("bert-base-uncased") 

    # -- run encoder
    outputs = encoder(tokens)

    # -- print outputs
    print(type(outputs)) # Class BaseModelOutputWithPastAndCrossAttentions
    print(outputs.last_hidden_state)
    print(outputs.hidden_states)
    print(outputs.attentions)
    print(outputs.cross_attentions)

