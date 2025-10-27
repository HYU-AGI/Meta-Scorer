# This code is based on the original repository by Junyi Ye (MIT License, 2024)
# Source: https://github.com/JunyiYe/CreativeMath


from models.api_models import generate_api_response, load_api_model
from models.local_models import generate_local_response, load_local_model
from models.prompt_utils import load_messages


class ModelWrapper:
    def __init__(self, args):
        self.args = args
        self.model_name = self.args.model_name
        self.is_api_model = self.model_name in ["gemini-1.5-pro", "o4-mini"]

        if self.is_api_model:
            self.model = load_api_model(self.args)
        else:
            self.model, self.tokenizer = load_local_model(self.args)

    def generate_response(self, prompt):
        messages = load_messages(self.model_name, prompt)
        if self.is_api_model:
            return generate_api_response(self.model_name, self.model, messages, self.args)
        else:
            return generate_local_response(self.model, self.tokenizer, messages, self.args)
