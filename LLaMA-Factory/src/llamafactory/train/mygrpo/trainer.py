# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
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

import math
import os
import sys
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional
import inspect
import torch
import copy
import editdistance

from accelerate.utils import gather_object, gather, DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override
from contextlib import nullcontext
from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from torch.utils.data import DataLoader, Sampler
from collections import defaultdict, deque


from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    ProcessorMixin,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments
from .utils import (
    RepeatSampler,
    pad,
    selective_log_softmax,
    split_tensor_dict, 
    shuffle_sequence_dict,
    nanstd,
    entropy_from_logits,
    nanmin,
    nanmax
)
from functools import partial
from transformers.trainer_utils import seed_worker






logger = logging.get_logger(__name__)

class CustomGRPOTrainer(Trainer):
    def __init__(
        self,
        model: Any,
        model_args: "ModelArguments",
        training_args: Any,
        train_dataset: Any,
        tokenizer: Any,
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[list["TrainerCallback"]],
        ref_model: Optional[Any],
        reward_model: Optional[Any],
        data_collator: Any,
        processor: Optional[Any],
        eval_dataset: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
            compute_loss_func="non-None value to disable scaling",
            **kwargs
        )

        self.model_accepts_loss_kwargs = False
        self.training_args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.ref_model = ref_model
        self.generating_args = generating_args
        
        # GRPO 超参数
        
        # 生成配置
        # self.generation_config = GenerationConfig.from_pretrained("")
        self.generation_config.update(**generating_args.to_dict())
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config.eos_token_id = [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids
        # self.generation_config = GenerationConfig(
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
        #     **generating_args.to_dict(),
        # )
        


        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # # print(generating_args)
        # self.generation_config = GenerationConfig(
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     eos_token_id=[self.tokenizer.eos_token_id],
        #     **generating_args.to_dict(),
        # )
        # # print(self.generation_config.to_dict())
        
        # 如果是 Peft 模型且没有显式 ref_model，我们将使用 disable_adapter

        self._buffered_inputs = None
        self._step = 0

        self.beta = self.finetuning_args.grpo_beta # KL penalty coefficient, 可以加到 hparams 里

        self.epsilon = self.finetuning_args.grpo_epsilon # Clip range for PPO-like loss inside GRPO
        self.importance_sampling_level = self.finetuning_args.grpo_importance_sampling_level
        # # print("self.beta")
        # # print(self.beta)
        # # print("self.epsilon")
        # # print(self.epsilon)
        # # print("self.importance_sampling_level")
        # # print(self.importance_sampling_level)


        self.mask_truncated_completions = False

        self.shuffle_dataset = self.finetuning_args.grpo_shuffle_dataset
        self.num_iterations = self.finetuning_args.grpo_num_iterations
        self.num_generations =  self.finetuning_args.grpo_num_generations
        self.steps_per_generation = training_args.gradient_accumulation_steps
        self.generation_batch_size = training_args.per_device_train_batch_size * self.steps_per_generation


        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )

        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )

        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        # # print("self.is_encoder_decoder", self.is_encoder_decoder)

        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # # print("self.model_kwarg_keys")
        # # print(self.model_kwarg_keys)

        self.temperature = generating_args.temperature
        # # print("self.temperature")
        # # print(self.temperature)

        self.reward_weights = torch.ones(1, dtype=torch.float32)

        self.scale_rewards = "group"

        self._logs = {
            "prompt": deque(maxlen=self.generation_batch_size),
            "completion": deque(maxlen=self.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=self.generation_batch_size)),
            "advantages": deque(maxlen=self.generation_batch_size),
        }

        self.loss_type = "grpo"
        self.top_entropy_quantile = 1.0

        self.off_policy_mask_threshold = None
        self.use_bias_correction_kl = False

        self.epsilon_low = self.epsilon
        self.epsilon_high = self.epsilon

        self.log_completions = False
    
    
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        # print("self._train_batch_size")
        # print(self._train_batch_size)
        dataloader_params = {
            "batch_size": self._train_batch_size * self.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.training_args.dataloader_num_workers,
            "pin_memory": self.training_args.dataloader_pin_memory,
            "persistent_workers": self.training_args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.training_args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker
            )

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.training_args.seed,
        )
    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations_eval,
            seed=self.training_args.seed,
        )
    

    def training_step(self, model, inputs, num_items_in_batch):
        # print("inputs in training_step")
        # print(inputs)
        output = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        return output

    def _prepare_inputs(self, generation_batch):
        mode = "train"
        if mode == "train":
            generate_every = self.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.steps_per_generation)
                self._buffered_inputs = [batch for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.steps_per_generation]
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        # print("after _prepare_inputs")
        # print(inputs)
        return inputs

    def compute_custom_rewards(self, messages: list[str], labels: list[str]) -> list[torch.Tensor]:
        """
        计算混合奖励：WER (词) + CER (字符) + SER (BBPE Token)
        """
        # # print("----------get_rewards_from_server labels-----------")
        # responses = [text.replace('\n', '') for text in messages]
        # # print(responses)
        # ground_truth = [text.replace('\n', '') for text in labels]
        # # print(ground_truth)
        w_word = 0.3
        w_char = 0.3
        w_token = 0.4 
        sharpness = 1.0 
        rewards = []
        for pred_text, label_text in zip(messages, labels):
            
            pred_clean = pred_text.replace('\n', '').strip().lower()
            label_clean = label_text.replace('\n', '').strip().lower()
            # print("compute_custom_rewards")
            # print(pred_clean)
            # print(label_clean)
            if not label_clean:
                rewards.append(torch.tensor(0.0 if pred_clean else 1.0))
                raise ValueError("not label_clean")
                continue
            dist_char = editdistance.eval(pred_clean, label_clean)
            len_char = max(len(pred_clean), len(label_clean))
            score_char = 1.0 - (dist_char / len_char) if len_char > 0 else 0.0
            pred_words = pred_clean.split()
            label_words = label_clean.split()
            dist_word = editdistance.eval(pred_words, label_words)
            len_word = max(len(pred_words), len(label_words))
            score_word = 1.0 - (dist_word / len_word) if len_word > 0 else 0.0

            pred_ids = self.tokenizer.encode(pred_clean, add_special_tokens=False)
            label_ids = self.tokenizer.encode(label_clean, add_special_tokens=False)
            
            dist_token = editdistance.eval(pred_ids, label_ids)
            len_token = max(len(pred_ids), len(label_ids))
            score_token = 1.0 - (dist_token / len_token) if len_token > 0 else 0.0

            raw_score = (w_word * score_word) + (w_char * score_char) + (w_token * score_token)
            
            shaped_score = raw_score ** sharpness

            final_reward = shaped_score

            if dist_char == 0:
                final_reward += 0.5
            
            rewards.append(torch.tensor(final_reward, dtype=torch.float32))

        return rewards




    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list, ground_truth):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), 1, device=device)
        # print("_calculate_rewards")
        # print(inputs)
        # print(prompts)
        # print(completions)
        # print(completion_ids_list)
        # print(ground_truth)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        # keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        # reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        output_reward_func = self.compute_custom_rewards(completions, ground_truth)
        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
        rewards_per_func[:, 0] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)


        # If all reward functions return None for a given row, issue a detailed warning
        # if torch.isnan(rewards_per_func).all(dim=1).any():
        #     nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
        #     row_reward_kwargs = {
        #         key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
        #     }
        #     row_reward_kwargs["prompt"] = prompts[nan_row_idx]
        #     row_reward_kwargs["completion"] = completions[nan_row_idx]
        #     logger.warning(
        #         f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
        #         "Please ensure that at least one reward function returns a valid reward."
        #     )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func



    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        feature_attention_mask = None,
        input_features = None,
    ) -> dict[str, torch.Tensor | None]:
        # print("_get_per_token_logps_and_entropies")
        # print(input_ids)
        # print(input_ids.shape)
        # print(attention_mask)
        # print(attention_mask.shape)
        # print(logits_to_keep)
        # print(batch_size)
        # print(feature_attention_mask)
        # print(feature_attention_mask.shape)
        # print(input_features)
        # print(input_features.shape)
        # raise ValueError("not complete")
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]
            input_features_batch = input_features[start : start + batch_size]
            feature_attention_mask_batch = feature_attention_mask[start : start + batch_size]
            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch, "input_features": input_features_batch, "feature_attention_mask": feature_attention_mask_batch}
            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # print("logits_to_keep in self.model_kwarg_keys")
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            # print("after logits_to_keep in self.model_kwarg_keys")

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

            logits = model(**model_inputs).logits
            # print("model(**model_inputs).logits")
            # print(logits)
            # print(logits.shape)
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            # print("after process")
            pred_ids = torch.argmax(logits, dim=-1)
            text_list = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            # print("pred_ids")
            # print(pred_ids)
            # print(text_list)




            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            # print("completion_ids")
            # print(completion_ids)
            # print(self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True))



            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies




    def _generate_single_turn(self, inputs: list):
        device = self.accelerator.device
        mode = "train"

        # Generate completions using either vLLM or regular generation
        if False and self.use_vllm:
            raise ValueError("self.use_vllm")
        elif False and self.use_transformers_paged:
            raise ValueError("self.use_transformers_paged")
        else:
            
            generate_inputs = super()._prepare_inputs(inputs)

            grround_labels = generate_inputs.pop('labels', None)
            # print("generate_inputs")
            # print(generate_inputs)
            # print("grround_labels")
            # print(grround_labels)

            

            with (unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model, torch.no_grad(), nullcontext()):
                unwrapped_model= self.accelerator.unwrap_model(self.model)
                prompt_completion_ids: torch.Tensor = unwrapped_model.generate(
                    generation_config=self.generation_config,temperature=self.generating_args.temperature,top_p=self.generating_args.top_p,repetition_penalty=self.generating_args.repetition_penalty,logits_processor=get_logits_processor(), **generate_inputs
                )
            
            

            # Compute prompt length and extract completion ids
            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            # print(completion_ids)

            completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            # print("init_completions")
            # print(completions)
            # ground_truth = self.tokenizer.batch_decode(grround_labels, skip_special_tokens=True)
            # # print(ground_truth)
            # raise ValueError("not complete")

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
            completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]
            logprobs = None  # not used in this case
            extra_fields = {}  
            # print(prompt_ids)
            # print(completion_ids)
            # print(logprobs)
            # print(extra_fields)
            # print(completion_mask)
            completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            # print(completions)
            # print(self.pad_token_id)
            # print(self.eos_token_id)


        return prompt_ids, completion_ids, logprobs, extra_fields

    def _generate(self, inputs: list):
        device = self.accelerator.device
        mode = "train"

        # Copy the prompts to avoid modifying the original list
        inputs = copy.deepcopy(inputs)

        prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(inputs)
        

        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        # print("_generate completions")
        # print(completions)

        # raise ValueError("not complete")

        # Extract tool calls from the completions and (possibly) execute them

        tool_mask = None

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        # print("prompt_lengths")
        # print(prompt_lengths)

        if tool_mask is not None:  # count only non-tool tokens (tool_mask=1)
            raise ValueError("tool_mask is not None")
        else:
            completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        # print("completion_lengths")
        # print(completion_lengths)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()  # = num_items_in_batch, required for the DAPO loss

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return (
            prompt_ids,
            completion_ids,
            tool_mask,
            completions,
            total_completion_tokens,
            logprobs,
            extra_fields,
        )



    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        # print("_generate_and_score_completions")
        # print("inputs")
        # print(inputs)
        feature_attention_masks = inputs.get('feature_attention_mask', None)
        input_features = inputs.get('input_features', None)
        # print("extrated feature_attention_masks")
        # print(feature_attention_masks)
        # print("extrated input_features")
        # print(input_features)
        # raise ValueError("not complete")

        ground_labels = inputs.pop('labels', None)
        # print(ground_labels)
        # print(ground_labels[0])
        labels_to_decode = ground_labels.clone()
        labels_to_decode[labels_to_decode == -100] = self.pad_token_id
        ground_truth = self.tokenizer.batch_decode(labels_to_decode, skip_special_tokens=True)
        # print("batch_decode ground_truth")
        # print(ground_truth)

        mode = "train"

        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(inputs)
        # print("after self._generate(inputs)")
        # print(prompt_ids_list)
        # print(completion_ids_list)
        # print(completions)
        # print(num_items_in_batch)
        # print(sampling_per_token_logps_list)
        # print(extra_fields)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None
        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.training_args.per_device_train_batch_size if mode == "train" else self.per_device_eval_batch_size



        # print("************_generate_and_score_completions*************")
        # print(prompt_ids)
        # print(prompt_mask)
        # print(completion_ids)
        # print(completion_mask)
        # print(prompt_completion_ids)

        # print('\n\n')
        generate_every = self.steps_per_generation * self.num_iterations  # generation frequency
        # print(self.steps_per_generation)
        # print(self.num_iterations)
        # print(generate_every)
        # print(self.training_args.gradient_accumulation_steps)
        # print(self.training_args.gradient_accumulation_steps % generate_every)

        
        # self.training_args.gradient_accumulation_steps

        forward_kwargs = {}
        if input_features is not None:
            forward_kwargs["input_features"] = input_features
        if feature_attention_masks is not None:
            forward_kwargs["feature_attention_mask"] = feature_attention_masks

        # print("forward_kwargs[feature_attention_mask]")
        # print(forward_kwargs["feature_attention_mask"])

        with torch.no_grad():
            generate_every = self.steps_per_generation * self.num_iterations  # generation frequency
            if self.training_args.gradient_accumulation_steps % generate_every != 0 :
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None
            # print("old_per_token_logps")
            # print(old_per_token_logps)
            # print("ref_per_token_logps")
            # print(ref_per_token_logps)
        
        

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # print("prompts_text")
        # print(prompts_text)
        # print("completions_text")
        # print(completions_text)


        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values
        

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompt_ids, completions, completion_ids_list, ground_truth)
        # print("rewards_per_func", rewards_per_func)
        
        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # print("rewards")
        # print(rewards)
        # raise ValueError("not complete")

        # Compute grouped-wise rewards
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            if num_generations > 1:
                std_rewards = rewards.view(-1, num_generations).std(dim=1)
                std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
            else:  # this case doesn't occur during training, but could in eval when num_generations_eval=1
                std_rewards = torch.zeros_like(rewards)
        elif self.scale_rewards == "batch":
            # Compute global std
            if rewards.numel() > 1:
                std_rewards = rewards.std().expand_as(rewards)
            else:  # this case doesn't occur during training, but could in eval when num_generations_eval=batch_size=1
                std_rewards = torch.zeros_like(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompt_ids),
            (self.accelerator.process_index + 1) * len(prompt_ids),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]
        # print("advantages")
        # print(advantages)

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        # for i, reward_func_name in enumerate(self.reward_func_names):
        mean_rewards = torch.nanmean(rewards_per_func[:, 0]).item()
        self._metrics[mode][f"rewards/custom_rewards/mean"].append(mean_rewards)
        std_func_rewards = nanstd(rewards_per_func[:, 0]).item()
        self._metrics[mode][f"rewards/custom_rewards/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        # for i, name in enumerate(self.reward_func_names):
        self._logs["rewards"]['custom_rewards'].extend(rewards_per_func[:, 0].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if "input_features"  in forward_kwargs:
            output["input_features"] = forward_kwargs["input_features"]
        if "feature_attention_mask" in forward_kwargs:
            output["feature_attention_mask"] = forward_kwargs["feature_attention_mask"]
        # print("output[feature_attention_mask]")
        # print(output["feature_attention_mask"])
        # print(output["feature_attention_mask"].shape)
        return output
    

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        # print("inputs in compute_loss")
        # print(inputs)
        # print(inputs["feature_attention_mask"])
        # print(inputs["feature_attention_mask"].shape)
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

        input_features, feature_attention_mask = inputs["input_features"], inputs["feature_attention_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            input_features = input_features,
            feature_attention_mask = feature_attention_mask,
            compute_entropy=True,
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        # In the base GRPO implementation, advantages are expected to have shape (B,). To support subclasses that
        # provide advantages with shape (B, T) (e.g., MiniLLM), we *conditionally* unsqueeze the tensor.
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        # print("advantages", advantages)
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        if self.off_policy_mask_threshold is not None:
            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                old_per_token_logps=old_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            # Importance sampling correction for the KL divergence
            if self.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            # if self.args.delta is not None:
            #     coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            per_token_loss = torch.empty_like(coef_1)
            positive_advantages_mask = advantages.repeat([1, coef_1.shape[1]]) > 0
            per_token_loss[positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[positive_advantages_mask], self.args.sapo_temperature_pos
            )
            per_token_loss[~positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[~positive_advantages_mask], self.args.sapo_temperature_neg
            )
            per_token_loss = -per_token_loss * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())
        # print("loss", loss)
        return loss
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            logging_backends = []
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                logging_backends.append(wandb)
            if self.args.report_to and "trackio" in self.args.report_to:
                logging_backends.append(trackio)

            table = {
                "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                "prompt": self._logs["prompt"],
                "completion": self._logs["completion"],
                **self._logs["rewards"],
                "advantage": self._logs["advantages"],
            }

            df_base = pd.DataFrame(table)
            images_raw = self._logs["images"] or []

            for logging_backend in logging_backends:
                if images_raw:
                    images = []
                    for image_list in self._logs["images"]:
                        images.append([logging_backend.Image(image) for image in image_list])
                    df = pd.concat(
                        [df_base, pd.Series(images, name="image")],
                        axis=1,
                        copy=False,
                    )
                else:
                    df = df_base

                if self.log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])

                logging_backend.log({"completions": logging_backend.Table(dataframe=df)})