from transformers import BertGenerationConfig, BertGenerationDecoder, BertGenerationEncoder, EncoderDecoderModel, BertTokenizer

# -- This model can be trained according to hugging face transformer procedure, otherwise a custom decoder has to be tailored with custom training

if __name__=='__main__':

    # -- Initializing (pretrained) Encoder
    encoder = BertGenerationEncoder.from_pretrained("bert-base-uncased") 

    # -- Initializing (untrained) decoder BertGeneration config
    config = BertGenerationConfig()
    config.is_decoder=True
    config.add_cross_attention=True
    
    # -- Initializing a model (untrained) decoder from the config
    decoder = BertGenerationDecoder(config)

    # -- Combining models into a EncoderDecoderModel (transformers package)
    encoder_decoder = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    
    # -- Testing encoder_decoder 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text = "This is a test sentence, let's see how it works"
    tokens = tokenizer(text, return_tensors="pt").input_ids

    outputs = encoder_decoder.generate(tokens)

    print(tokenizer.decode(outputs[0])) # should output random text as it is not trained
