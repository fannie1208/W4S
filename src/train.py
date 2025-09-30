# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import dataclasses
import inspect
import json
import os
import pathlib
import torch
import torch.nn as nn
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Any, Union, Callable
from torch.utils.data import Dataset
from transformers.utils import is_peft_available, is_liger_kernel_available
import transformers
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    DataCollatorForLanguageModeling,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils.deprecation import deprecate_kwarg
from transformers.trainer_pt_utils import LabelSmoother
from fastchat.model.model_adapter import get_conversation_template
from trl import (
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM

if is_wandb_available():
    import wandb

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )


@dataclass
class RLConfig(TrainingArguments):
    r"""
    Configuration class for the [`WeightedTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.
    """

    dataset_text_field: str = field(
        default="text",
        metadata={
            "help": "Name of the text field of the dataset. If provided, the trainer will automatically create a "
            "`ConstantLengthDataset` based on `dataset_text_field`."
        },
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Controls whether the `ConstantLengthDataset` packs the sequences of the dataset."},
    )
    learning_rate: float = field(
        default=2.0e-5,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`TrainingArguments`."
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum sequence length for the `ConstantLengthDataset` and for automatically creating the "
            "dataset. If `None`, it uses the smaller value between `tokenizer.model_max_length` and `1024`."
        },
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset. Only used when `packing=False`."},
    )
    dataset_batch_size: int = field(
        default=1000,
        metadata={
            "help": "Number of examples to tokenize per batch. If `dataset_batch_size <= 0` or `dataset_batch_size is "
            "None`, tokenizes the full dataset as a single batch."
        },
    )
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model "
            "from a string."
        },
    )
    dataset_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Dictionary of optional keyword arguments to pass when creating packed or non-packed datasets."
        },
    )
    eval_packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to pack the eval dataset. If `None`, uses the same value as `packing`."},
    )
    num_of_sequences: int = field(
        default=1024,
        metadata={"help": "Number of sequences to use for the `ConstantLengthDataset`."},
    )
    chars_per_token: float = field(
        default=3.6, metadata={"help": "Number of characters per token to use for the `ConstantLengthDataset`."}
    )
    use_liger: bool = field(
        default=False,
        metadata={"help": "Monkey patch the model with Liger kernels to increase throughput and reduce memory usage."},
    )


class WeightedTrainer(Trainer):
    _tag_names = ["trl", "rl"]

    @deprecate_kwarg(
        "tokenizer", "0.16.0", "processing_class", warn_if_greater_or_equal_version=True, raise_if_both_names=True
    )
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[RLConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable] = None,
    ):
        if args is None:
            args = RLConfig(output_dir="tmp_trainer")
        elif args is not None and args.__class__.__name__ == "TrainingArguments":
            args_as_dict = args.to_dict()
            # Manually copy token values as TrainingArguments.to_dict() redacts them
            args_as_dict.update({k: getattr(args, k) for k in args_as_dict.keys() if k.endswith("_token")})
            args = RLConfig(**args_as_dict)

        if getattr(args, "model_init_kwargs", None) is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_init_kwargs to the RLConfig, but your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the RLConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            if args.use_liger:
                model = AutoLigerKernelForCausalLM.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if args.packing and data_collator is not None and isinstance(data_collator, DataCollatorForCompletionOnlyLM):
            raise ValueError(
                "You passed a `DataCollatorForCompletionOnlyLM` to the WeightedTrainer. This is not compatible with the `packing` argument."
            )

        if is_peft_available() and peft_config is not None:
            if not isinstance(peft_config, PeftConfig):
                raise ValueError(
                    "If you want to use the PeftModel, you need to pass a PeftConfig object to the WeightedTrainer."
                    f" and you passed a {type(peft_config)}."
                )

            if not isinstance(model, PeftModel):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )
                gradient_checkpointing_kwargs = getattr(args, "gradient_checkpointing_kwargs", None) or {}
                is_sharded_qlora = False
                # Below is to support QLoRA + FSDP / DS-Zero3 - one should never call
                # peft_module_casting_to_bf16 or prepare_model_for_kbit_training when doing
                # QLoRA + FSDP / DS-Zero3
                if getattr(model, "is_loaded_in_4bit", False):
                    for _, param in model.named_parameters():
                        if param.__class__.__name__ == "Params4bit":
                            is_sharded_qlora = param.data.device.type in {"cpu", "meta"}
                            break
                if getattr(model, "is_loaded_in_8bit", False) or (
                    getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora
                ):
                    prepare_model_kwargs = {
                        "use_gradient_checkpointing": getattr(args, "gradient_checkpointing", False)
                    }

                    if _support_gc_kwargs:
                        prepare_model_kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

                    if args is not None:
                        args = dataclasses.replace(args, gradient_checkpointing=False)
                elif getattr(args, "gradient_checkpointing", False) and (
                    "use_reentrant" not in gradient_checkpointing_kwargs
                    or gradient_checkpointing_kwargs["use_reentrant"]
                ):
                    # For backward compatibility with older versions of transformers
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    else:

                        def make_inputs_require_grad(module, input, output):
                            output.requires_grad_(True)

                        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

                if (
                    "autocast_adapter_dtype" in list(inspect.signature(get_peft_model).parameters)
                    and getattr(model, "is_loaded_in_4bit", False)
                    and is_sharded_qlora
                ):
                    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
                else:
                    model = get_peft_model(model, peft_config)
                if (
                    args is not None
                    and args.bf16
                    and getattr(model, "is_loaded_in_4bit", False)
                    and not is_sharded_qlora
                ):
                    peft_module_casting_to_bf16(model)

        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(processing_class, "pad_token", None) is None:
                processing_class.pad_token = processing_class.eos_token

        if args.max_seq_length is None:
            # to overcome some issues with broken tokenizers
            args.max_seq_length = min(processing_class.model_max_length, 1024)

        self.dataset_num_proc = args.dataset_num_proc
        self.dataset_batch_size = args.dataset_batch_size

        if args.dataset_kwargs is None:
            args.dataset_kwargs = {}

        # if formatting_func is None:
        #     # check if dataset has ChatML format or instruction format and is supported
        #     # if not stays None
        #     formatting_func = get_formatting_func_from_dataset(train_dataset, processing_class)
        #     # if a template is detected, we don't need to add special tokens again
        #     if formatting_func is not None:
        #         args.dataset_kwargs["add_special_tokens"] = False

        if not args.packing:
            if data_collator is None:
                data_collator = DataCollatorForLanguageModeling(tokenizer=processing_class, mlm=False)
        
        if processing_class.padding_side is not None and processing_class.padding_side != "right":
            warnings.warn(
                "You passed a processing_class with `padding_side` not equal to `right` to the RLTrainer. This might "
                "lead to some unexpected behaviour due to overflow issues when training a model in half-precision. "
                "You might consider adding `processing_class.padding_side = 'right'` to your code.",
                UserWarning,
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if self.train_dataset is not None:
            if self.args.max_steps > 0 and args.packing:
                self.train_dataset.infinite = True
            elif self.args.max_steps == -1 and args.packing:
                self.train_dataset.infinite = False
    
    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="RL",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
    
    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        return data_collator

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        reward_weights = inputs.pop("reward_weights")
        for key in inputs.keys():
            num_of_samples = inputs[key].shape[0]
            break
        sample_loss = []
        sample_token = []
        for i in range(num_of_samples):
            sample = {}
            for key in inputs.keys():
                sample[key] = inputs[key][i:i+1]
            num_of_tokens = sum(inputs['attention_mask'][i])
            if return_outputs: 
                loss, outputs = super().compute_loss(model, sample, return_outputs)
            else:
                loss = super().compute_loss(model, sample, return_outputs)
            sample_loss.append(loss)
            sample_token.append(len(inputs['attention_mask'][i]))
        log = {
            "sample_loss": sample_loss
        }
        sample_loss_tensor = torch.stack(sample_loss).to(reward_weights.device)
        reward_weights_reshaped = reward_weights.view(-1)
        weighted_loss = sample_loss_tensor * reward_weights_reshaped
        average_weighted_loss = weighted_loss.mean()

        return (average_weighted_loss, outputs) if return_outputs else average_weighted_loss

class SFTTrainer(WeightedTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Remove reward_weights since we don't need them
        
        for key in inputs.keys():
            num_of_samples = inputs[key].shape[0]
            break

        sample_loss = []
        sample_token = []
        for i in range(num_of_samples):
            sample = {}
            for key in inputs.keys():
                sample[key] = inputs[key][i:i+1]
            num_of_tokens = sum(inputs['attention_mask'][i])
            if return_outputs:
                loss, outputs = super().compute_loss(model, sample, return_outputs)
            else:
                loss = super().compute_loss(model, sample, return_outputs)
            sample_loss.append(loss)
            sample_token.append(len(inputs['attention_mask'][i]))

        log = {
            "sample_loss": sample_loss
        }
        sample_loss_tensor = torch.stack(sample_loss)
        average_loss = sample_loss_tensor.mean()

        return (average_loss, outputs) if return_outputs else average_loss


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
    conv = get_conversation_template("qwen")

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    total_num_of_data = len(sources)
    valid_num_of_data = 0

    # Apply prompt templates
    conversations = []
    losses = []
    for i, source in enumerate(sources):
        conv.system_message = source[0]["value"].strip()
        conv.messages = []
        losses_for_one_interaction = []

        source = source[1:]
        
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # assert role == conv.roles[(j+1) % 2], f"{i}"
            conv.append_message(role, sentence["value"])
            losses_for_one_interaction.append(sentence["loss"])
        conversations.append(conv.get_prompt())
        losses.append(losses_for_one_interaction)

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    reward_weights = None
    # print(conversations[0])
    # Mask targets. Only compute loss on the assistant outputs.
    sep = "<|im_start|>assistant\n"
    for idx, (conversation, target) in enumerate(zip(conversations, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split("<|im_start|>user\n")
        cur_len = 0
        # cur_len = 1  # for <s> special character
        # target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            if i == 0:
                turn_len = len(tokenizer(turn).input_ids)
                target[cur_len: cur_len + turn_len] = IGNORE_TOKEN_ID
                cur_len += turn_len
                continue
            turn = f"<|im_start|>user\n{turn}"
            parts = turn.split(sep)  # user text and assistant text
            if len(parts) != 2:
                break
            parts[0] += sep
            turn_len = len(tokenizer(turn).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids)
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            if sources[idx][2*i]["loss"] == 0.0:
                target[cur_len + instruction_len : cur_len + turn_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID
        if reward_weights == None:
            reward_weights = [[sources[idx][-1]["loss"]]]
        else:
            reward_weights.append([sources[idx][-1]["loss"]])

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                    f"\n{conversation}"
                    f"\n{target}"
                )
            else:
                valid_num_of_data += 1

    print(f"Valid data portion: {valid_num_of_data}/{total_num_of_data}")

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        reward_weights=torch.tensor(reward_weights)
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.reward_weights = data_dict["reward_weights"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
            reward_weights=self.reward_weights[i]
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)
        
def main(data_args, training_args, model_args):
    global local_rank
    ################
    # Model init kwargs & Tokenizer
    ################
    local_rank = training_args.local_rank
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )
    # tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.pad_token
    ################
    # Dataset
    ################
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    ################
    # Training
    ################
    trainer = WeightedTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        **data_module,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    # trainer = SFTTrainer(
    #     model=model_args.model_name_or_path,
    #     args=training_args,
    #     **data_module,
    #     processing_class=tokenizer,
    #     peft_config=get_peft_config(model_args),
    # )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (DataArguments, RLConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("rl", help="Run the RL training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    data_args, training_args, model_args = parser.parse_args_and_config()
    main(data_args, training_args, model_args)