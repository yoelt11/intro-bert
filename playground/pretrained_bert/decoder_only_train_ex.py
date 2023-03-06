from torch.nn import functional as F
from transformers import AdamW
from transformers import BertGenerationConfig, BertGenerationDecoder, BertGenerationEncoder, EncoderDecoderModel, BertTokenizer

# -- This model can be trained according to hugging face transformer procedure, otherwise a custom decoder has to be tailored with custom training
# -- Check: https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing#scrollTo=ZwQIEhKOrJpl

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
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids']
    encoder_attention_mask = inputs['attention_mask']

    target_text = "This is a test"
    targets = tokenizer(target_text, return_tensors="pt", padding=True, truncation=True)
    decoder_input_ids = targets['input_ids']
    labels = targets['input_ids'].copy()
    decoder_attention_mask = targets['attention_mask']
    
    # --  Initialize Optimizer
    optimizer = AdamW(encoder_decoder.decoder.parameters(), lr=1e-5) # decoder only being optimized

    # -- components of trianing lopp
    for param in encoder_decoder.encoder.parameters(): # only to calculate gradients for decoder
        param.requires_grad = False 
    outputs = encoder_decoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss}") # should output random text as it is not trained
