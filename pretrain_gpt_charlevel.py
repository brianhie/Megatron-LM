from pretrain_gpt import (
    pretrain,
    train_valid_test_datasets_provider,
    model_provider,
    ModelType,
    forward_step,
)

if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'CharLevelTokenizer'})
