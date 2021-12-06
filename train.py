import os
import pandas as pd
import numpy as np
import torch
from transformers import get_scheduler, AdamW
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer
from transformers import default_data_collator
from argparse import ArgumentParser
#from tqdm import tqdm
import time, datetime
import random
import datasets
import collections
from functools import partial

#torch.cuda.empty_cache()

def seed_everything(seed=772):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

"""# Args"""

def parse_args():
	parser = ArgumentParser(description='Word Meaning Comparison')
    
	parser.add_argument('--data_paths', '-d', type=str, default=[
        'dataset/train.csv',
        'dataset/mlqa_hindi.csv',
        'dataset/xquad.csv',
        'dataset/squad_translated_tamil.csv'
    ])

	parser.add_argument('--model', '-m', type=str, choices=['mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp', 'mrm8488/bert-multi-cased-finetuned-xquadv1', 'alon-albalak/xlm-roberta-base-xquad', 'deepset/xlm-roberta-large-squad2'], 
                        
                default='alon-albalak/xlm-roberta-large-xquad',
    )

	parser.add_argument('--out_dir', type=str, default='out')
	parser.add_argument('--models_dir', type=str, default='models')
	parser.add_argument('--tmp_dir', type=str, default='tmp')
	
	parser.add_argument('--gpu', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--epochs', type=int, default=3)
	parser.add_argument('--lr', type=float, default=2e-5)

	parser.add_argument('--max_len', type=int, default=384)
	parser.add_argument('--doc_stride', type=int, default=128)
	parser.add_argument('--max_answer_length', type=int, default=30)
	parser.add_argument('--best_answer_length', type=int, default=20)

	return parser.parse_known_args()[0]
args = parse_args()
print(args)

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def convert_answers(r):
    start = r[0]
    text = r[1]
    return {"answer_start": [start], "text": [text]}

# ref: https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb
def prepare_train_features(examples, tokenizer, pad_on_right, max_length, doc_stride):
    examples["question"] = [q.lstrip() for q in examples["question"]]
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples



# ref: https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb
def prepare_validation_features(examples, tokenizer, pad_on_right, max_length, doc_stride):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples
tokenizer = AutoTokenizer.from_pretrained(args.model)
def process_data(data_paths):
    seed_everything()
    usecols = ["context", "question", "answer_text", "answer_start"]
    df = pd.read_csv(data_paths[0], usecols=usecols)
    for data_path in data_paths[1:]:
        df = pd.concat([df, 
                        pd.read_csv(data_path, usecols=usecols)], 
                       axis=0).reset_index(drop=True)
    
    ###### df = df.sample(frac=1).reset_index(drop=True)
    
    df["answers"] = df[["answer_start", "answer_text"]].apply(convert_answers, axis=1)
    
    
    data = datasets.Dataset.from_pandas(df)
    data = data.train_test_split(test_size=0.1, shuffle=True)
    print(data)
    #print(data['train'][0])
    
    features = data.map(
        partial(
            prepare_train_features,
            tokenizer=tokenizer,
            pad_on_right=(tokenizer.padding_side == "right"),
            max_length=args.max_len,
            doc_stride=args.doc_stride,
        ),
        batched=True,
        remove_columns=data["train"].column_names,
    )
    
    return features

features = process_data(args.data_paths)

model = AutoModelForQuestionAnswering.from_pretrained(args.model)

model_name = args.model.split("/")[-1]
print(model_name)

data_collator = default_data_collator



train_args = TrainingArguments(
    f"{model_name}",
    evaluation_strategy = "epoch",
    save_strategy = 'epoch',
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size*4,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    #push_to_hub=True,
    ##report_to="none",
)

trainer = Trainer(
    model,
    train_args,
    train_dataset=features["train"],
    eval_dataset=features["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()