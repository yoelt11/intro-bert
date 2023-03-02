import tokenizers
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, LineByLineTextDataset, BertModel, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

# Convert data into required format
dataset= LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = '/kaggle/working/pretraining_data.txt',
    block_size = 128
)

print('No. of lines: ', len(dataset))

# Defining configuration of BERT for pretraining
config = BertConfig(
    vocab_size=50000,
    hidden_size=768, 
    num_hidden_layers=6, 
    num_attention_heads=12,
    max_position_embeddings=512
)
 
model = BertForMaskedLM(config)
print('No of parameters: ', model.num_parameters())


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Defining training configuration\
training_args = TrainingArguments(
    output_dir='/kaggle/working/',
    overwrite_output_dir=True,
    num_train_epochs=7,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Perfrom pre-training and save the model
trainer.train()
trainer.save_model('/kaggle/working/')
