import tokenizers

if __name__=="__main__":

    # -- initializer tokenizer
    bert_tokenizer  = tokenizers.BertWordPieceTokenizer()

    # -- file path of vocabulary file
    ''' The format of this file should be one setence per line '''
    filepath = "./pretraining_data.txt"

    # -- train tokenizer
    bert_tokenizer.train(
            files=[filepath],
            vocab_size=50000,
            min_frequency=3,
            limit_alphabet=1000
            )

    # -- save model
    bert_tokenizer.save_model('./')

