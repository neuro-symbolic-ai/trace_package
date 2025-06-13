from transformers import PretrainedConfig

class TextDatasetConfig(PretrainedConfig):
    model_type = "text_dataset"

    def __init__(self, max_length=128, model_type="encoder-decoder", **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.model_type = model_type
