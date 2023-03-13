from torch.nn import functional as F
from transformers import AdamW
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

    # -- Input (Input and Target setences are tokenized toguether)
    input_text = "This is a test sentence, let's see how it works"
    target_text = "This is a test"
    text_batch = [input_text, target_text]
    encodings = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = encodings['labels']
    
    # --  Initialize Optimizer
    optimizer = AdamW(encoder_decoder.parameters(), lr=1e-5)

    # -- components of trianing lopp
    outputs = encoder_decoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss}") # should output random text as it is not trained
