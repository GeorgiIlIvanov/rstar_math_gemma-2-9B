# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/meta-math/MetaMath
import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import io
from tqdm import tqdm
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import argparse
import json
import random;random.seed(42)
import numpy as np

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    transformers.set_seed(seed)
    torch.use_deterministic_algorithms(True)

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "<|user|>:\n{instruction}\n<|assistant|>: Let's think step by step and solve the problem with code."
    ),

}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    attn_impl: Optional[str] = field(default="eager") # flash_attention_2 \ sdpa \ eager


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    overwrite_output_dir: bool = field(default=True)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


# def preprocess(
#     sources: Sequence[str],
#     targets: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     """Preprocess the data by tokenizing."""
#     examples = [s + t for s, t in zip(sources, targets)]
#     examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
#     input_ids = examples_tokenized["input_ids"]
#     labels = copy.deepcopy(input_ids)
#     for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
#         label[:source_len] = IGNORE_INDEX
#     return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        data_path = data_args.data_path
        try:
            data_path = data_path
        except:
            # data_path = data_path
            pass
        try:
            list_data_dict = jload(data_path)
        except BaseException:
            with open(data_path, 'r') as f:
                lines = f.readlines()
            list_data_dict = [json.loads(line.strip()) for line in lines]

        list_data_dict = random.sample(list_data_dict,  len(list_data_dict))
        # print('actual length', len(list_data_dict))
        list_data_dict = list_data_dict[:data_args.data_length]
        # print('use length', len(list_data_dict))

        # logging.warning("Formatting inputs...")
        #correction 2 start
        logging.warning("Formatting inputs for Gemma-2 chat template...")
        self.full_texts = []
        self.source_lenghts = [] #Store length of user turn + model prompt for masking

        instruction_key = 'instruction' if 'instruction' in list_data_dict[0] else 'query'
        output_key = 'output' if 'output' in list_data_dict[0] else 'response'
        print(f"Using keys: `{instruction_key}` and `{output_key}` for SFT data") #Added print verificaton

        for example in tqdm(list_data_dict, desc="Formatting Data"):
            user_content = example.get(instruction_key, "")
            model_content = example.get(output_key, "")
            if not user_content or not model_content:
                logging.warning(f"Skipping example due to missing content: {example}")
                continue

        # 1. Construct the part to be masked (user turn + the trigger for model response)
        # `<start_of_turn>model\n` is the prompt for the model to start generating
        prompt_part = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"

        # 2. construct the full text sequence for tokenization including the model's response and EOS

        full_text = f"{prompt_part}{model_content}{tokenizer.eos_token}"
        self.full_texts.append(full_text)

        # 3. Tokenize only the prompt part to get its length for masking labels later
        prompt_part_tokens = tokenizer(prompt_part, add_special_tokens = False)["input_ids"]
        self.source_lenghts.append(len(prompt_part_tokens))
        logging.warning(f"Formated {len(self.full_texts)} samples.")
        if not self.full_texts:
            raise ValueError("No data sanples were successfully formatted,. Check data keys and conent.")

        # prompt_input = PROMPT_DICT["prompt_input"]
        # # print(list_data_dict[0])
        # if 'instruction' in list_data_dict[0]:
        #     pass
        # else:
        #     def get_input(query):
        #         if query.find('\n') == -1:
        #             return ''
        #         return '\n'.join(query.split('\n')[1:])
        #     list_data_dict = [{'instruction':data['query'], 'output':data['response']} for data in list_data_dict]
        # # import ipdb; ipdb.set_trace()
        # sources = [
        #     prompt_input.format_map(example)
        #     for example in list_data_dict 
        # ]
        # sources = []
        # for example in list_data_dict:
        #     if example['instruction'] == '':
        #         sources.append('')
        #     else:
        #         sources.append(prompt_input.format_map(example))
        # targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        # self.sources = sources
        # self.targets = targets 

# correction 2 end
    def __len__(self):
    #     return len(self.sources)
            return(self.full_texts)


    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
    #     return dict(input_ids=self.sources[i], labels=self.targets[i])
        return dict(text=self.full_texts[i], source_len = self.source_lenghts[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #Extract text and source lenghts from instances provided by __getitem__
        texts = [instance ['text'] for instance in instances]
        source_lenghts = [instance['source_len'] for instance in instances]

        data_dict = self.tokenizer(
            texts,
            return_tensors = "pt",
            padding = "longest",
            max_length = self.tokenizer.model_max_lenght,
            truncation =True,
            )

        input_ids = data_dict["input_ids"]
        labels = copy.deepcopy(input_ids)

        for i, source_len in enumerate(source_lenghts):
            mask_len = min(source_len, labels[i].size(0))
            labels[i, :mask_len] = IGNORE_INDEX

        labels[input_ids == self.tokenizer.pad_token_id] = IGNORE_INDEX

        data_dict["labels"] = labels

        return data_dict

        # sources = []
        # targets = []
        # for instance in instances:
        #     source = instance['input_ids']
        #     target = instance['labels']
        #     sources.append(source)
        #     targets.append(target)

        # data_dict = preprocess(sources, targets, self.tokenizer)
        # input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        # )
        # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # return dict(
        #     input_ids=input_ids,
        #     labels=labels,
        #     attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        # )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    data_args.data_length = int(remaining_args[1])
    print(training_args.run_name)

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    training_args.seed = seed

    # https://github.com/huggingface/transformers/issues/31787  attn_implementation=flash_attention_2 \ sdpa \ eager
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        attn_implementation=model_args.attn_impl,
        torch_dtype=torch.bfloat16 if model_args.attn_impl == "flash_attention_2" else torch.float32,
        use_cache = False,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side= "right",
        trust_remote_code=True,
    )

    # if tokenizer.pad_token is None:
    #     smart_tokenizer_and_embedding_resize(
    #         special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN), # DEFAULT_PAD_TOKEN is "[PAD]"
    #         tokenizer=tokenizer,
    #         model=model,
    #     )

    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Setting pad_token to eos_token for Gemma-2")
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens(
            {
                "additional_special_tokens": ['<code>', '<end_of_step>', '<end_of_code>', '<output>', '<end_of_output>', '<answer>', '<end_of_answer>', '<|user|>', '<|assistant|>', '<refine>', '<end_of_refine>', '\n<|assistant|>', "<error_info>", "<end_of_error_info>", "<BACK>"]
            },
            replace_additional_special_tokens=False,
        )
    model.resize_token_embeddings(len(tokenizer))
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()  # resume
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()