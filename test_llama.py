import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch_dipu

import time
import json
import torch

import os
from llama import ModelArgs, Tokenizer, Transformer, LLaMA


class LLaMAInference:
    def __init__(self, llama_path, model, device_map="auto", **kwargs):


        params_file = os.path.join(llama_path, model, "params.json")
        tokenizer_path = os.path.join(llama_path, "tokenizer.model")

        assert os.path.exists(os.path.join(llama_path, model)), f"Model {model} does not exist"
        assert os.path.exists(params_file), f"Model {model} does not exist"
        assert os.path.exists(tokenizer_path), f"Missing tokenizer in {llama_path}"



        model_args = ModelArgs()

        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = self.tokenizer.n_words
        self.device = torch.device("dipu")

        self.model = Transformer(model_args).to(self.device)
        torch.set_default_tensor_type(torch.FloatTensor)



        self.generator = LLaMA(self.model, self.tokenizer)

    def generate(self, texts, temperature=0.8, top_p=0.95, max_length=256, repetition_penalty=1, stop_ids=None, stop_words=None):
        start_time = time.time()
        results, stats = self.generator.generate(
            texts,
            self.device,
            max_gen_len=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_ids=stop_ids,
            stop_words=stop_words
        )
        end_time = time.time()
        stats["total_seconds"] = end_time - start_time
        stats["tok/s"] = max(stats["num_generated_tokens"]) / stats["total_seconds"]
        return results, stats


if __name__ == '__main__':

    llama = LLaMAInference("/root/", "7B", max_batch_size=2)


    start_generation = time.time()
    print(llama.generate(["Chat:\nHuman: Hi i am an human\nAI:"], stop_ids=[13]))
    print(f"Inference took {time.time() - start_generation:.2f} seconds")
