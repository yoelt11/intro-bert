
from transformers import BertGenerationConfig, BertGenerationDecoder, BertGenerationEncoder, EncoderDecoderModel, BertTokenizer

# -- This model can be trained according to hugging face transformer procedure, otherwise a custom decoder has to be tailored with custom training

if __name__=='__main__':

    # -- Testing encoder_decoder 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # -- Initializing (pretrained) Encoder
    encoder = BertGenerationEncoder.from_pretrained("bert-base-uncased", bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.sep_token_id) 

    # -- Initializing (untrained) decoder BertGeneration config
    config = BertGenerationConfig(bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.sep_token_id)
    config.is_decoder=True
    config.add_cross_attention=True
    
    # -- Initializing a model (untrained) decoder from the config
    decoder = BertGenerationDecoder(config)

    # -- Combining models into a EncoderDecoderModel (transformers package)
    encoder_decoder = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    encoder_decoder.config.decoder_start_token_id = tokenizer.cls_token_id 
    encoder_decoder.config.pad_token_id = tokenizer.pad_token_id

    # -- Input
    text = "This is a test sentence, let's see how it works"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    # -- Output
    text = "This is a test sentence"
    labels = tokenizer(text, return_tensors="pt").input_ids

    loss = encoder_decoder(input_ids=input_ids, labels=labels).loss

    print(f"Loss: {loss}") # should output random text as it is not trained
