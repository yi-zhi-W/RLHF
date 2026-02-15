# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/ppo_trainer.py
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
import editdistance
from .uyghur_bpe import uyghur_bpe




import math
import os
import sys
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional

import torch
from accelerate.utils import DistributedDataParallelKwargs
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

from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


import re
import Levenshtein
def calculate_cer(reference, hypothesis):
    """计算字符错误率 (Character Error Rate)"""
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    # 移除多余空格，统一小写（如果适用）
    ref = reference.strip()
    hyp = hypothesis.strip()
    
    dist = Levenshtein.distance(ref, hyp)
    length = len(ref)
    
    return dist / length


class CustomPPOTrainer(PPOTrainer, Trainer):
    r"""Inherit PPOTrainer."""

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[list["TrainerCallback"]],
        model: "AutoModelForCausalLMWithValueHead",
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_collator: "DataCollatorWithPadding",
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
    ) -> None:
        if eval_dataset is not None:
            raise NotImplementedError("PPOTrainer does not support eval dataset yet.")

        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
        )

        # Add deepspeed config
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning_rank0("PPOTrainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        self.current_device = get_current_device()  # patch for deepspeed training

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )
        if self.args.max_steps > 0:
            logger.info_rank0("max_steps is given, it will override any value given in num_train_epochs")

        self.amp_context = torch.autocast(self.current_device.type)
        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if finetuning_args.reward_model_type == "full":
            if self.is_deepspeed_enabled:
                if not (
                    getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self._current_multimodal_features = {}


    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""Implement training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer."""
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        logger.info_rank0("***** Running training *****")
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info_rank0(
            f"  Total train batch size (w. parallel, buffer, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info_rank0(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps:,}")
        logger.info_rank0(f"  Num optimization epochs per batch = {self.finetuning_args.ppo_epochs:,}")
        logger.info_rank0(f"  Total training steps = {max_steps:,}")
        logger.info_rank0(f"  Number of trainable parameters = {count_parameters(self.model)[0]:,}")

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)
            # print("-------------batch-----------------")
            # print(batch)
            # Get inputs
            self.model.eval()
            self.tokenizer.padding_side = "right"  # change padding side
            queries, responses, rewards = [], [], []
            input_features, feature_attention_masks = [], []
            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                mini_batch = {}
                mini_labels = {}
                for key, value in batch.items():
                    if key=='labels':
                        mini_labels[key] = value[idx : idx + self.config.mini_batch_size]
                        continue
                    if isinstance(value, torch.Tensor):
                        # 对 Tensor 进行切片，注意 input_features 第一维也是 batch size，所以可以直接切
                        mini_batch[key] = value[idx : idx + self.config.mini_batch_size]
                    else:
                        raise ValueError("not isinstance(value, torch.Tensor)")
                
                # mini_batch = {
                #     "input_ids": batch["input_ids"][idx : idx + self.config.mini_batch_size],
                #     "attention_mask": batch["attention_mask"][idx : idx + self.config.mini_batch_size],
                # }
                # print("-------------labels-----------------")
                # print(batch['labels'][idx : idx + self.config.mini_batch_size])
                # print(batch['input_features'][idx : idx + self.config.mini_batch_size].shape)
                self._current_multimodal_features = mini_batch
                mini_batch_queries, mini_batch_responses, mini_batch_input_features, mini_batch_feature_attention_masks  = self.get_inputs(mini_batch)
                # print("mini_batch_input_features")
                # print(mini_batch_input_features)
                # print("mini_batch_feature_attention_masks")
                # print(mini_batch_feature_attention_masks)
                self._current_multimodal_features = mini_batch
                mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_responses, mini_labels)
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards.extend(mini_batch_rewards)
                input_features.extend(mini_batch_input_features)
                # print("----------------input_features.extend(mini_batch_input_features)-------------------")
                # print(input_features)
                feature_attention_masks.extend(mini_batch_feature_attention_masks)
            # print("---------before step----------------")
            # print("queries")
            # print(queries)
            # print("responses")
            # print(responses)
            # print("input_features")
            # print(input_features)
            # print("feature_attention_masks")
            # print(feature_attention_masks)
            # raise ValueError("not over")
            # Run PPO step
            self.model.train()
            stats = self.step(queries, responses, rewards, input_features = input_features, feature_attention_masks = feature_attention_masks)
            # print(stats)
            self.tokenizer.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except Exception:
                    logger.warning_rank0("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.callback_handler.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                )
                self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    @override
    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_inputs(self, batch: dict[str, "torch.Tensor"]) -> tuple[list["torch.Tensor"], list["torch.Tensor"]]:
        r"""Generate model's responses given queries."""
        if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            raise ValueError("batch[input_ids].size(0) == 1")
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                batch[k] = v[:, start_index:]

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)

            generate_output: torch.Tensor = unwrapped_model.generate(
                generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
            )
            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)

        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1) :].detach().cpu()
        input_feature = batch["input_features"].detach()
        feature_attention_mask = batch["feature_attention_mask"].detach().cpu()
        # print("-----------get_inputs---- input_feature----------")
        # print(input_feature)
        queries, responses = [], []
        input_features, feature_attention_masks = [], []
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_indexes) == 0:  # allow empty response
                response_length = 1
            elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:  # include eos token
                response_length = response_indexes[-1].item() + 2
            else:
                response_length = response_indexes[-1].item() + 1

            queries.append(query[i, query_start_index:])  # remove padding from left
            responses.append(response[i, :response_length])  # remove padding from right
            input_features.append(input_feature[i,:])
            feature_attention_masks.append(feature_attention_mask[i,:])
        # print("------after process---input_features----")
        # print(input_features)
        return queries, responses, input_features, feature_attention_masks

    # def compute_custom_rewards(self, messages: list[str], labels: list[str]) -> list[torch.Tensor]:
    #     """
    #     计算混合奖励：WER (词) + CER (字符) + SER (BBPE Token)
    #     """
    #     w_word = 0.3
    #     w_char = 0.3
    #     w_token = 0.4 

    #     sharpness = 3.0 
        
    #     rewards = []
        

    #     for pred_text, label_text in zip(messages, labels):

    #         pred_clean = pred_text.replace('\n', '').strip().lower()
    #         label_clean = label_text.replace('\n', '').strip().lower()

    #         if not label_clean:
    #             rewards.append(torch.tensor(0.0 if pred_clean else 1.0))
    #             raise ValueError("not label_clean")
    #             continue

    #         dist_char = editdistance.eval(pred_clean, label_clean)

    #         len_char = max(len(pred_clean), len(label_clean))
    #         score_char = 1.0 - (dist_char / len_char) if len_char > 0 else 0.0

    #         pred_words = pred_clean.split()
    #         label_words = label_clean.split()
    #         dist_word = editdistance.eval(pred_words, label_words)
    #         len_word = max(len(pred_words), len(label_words))
    #         score_word = 1.0 - (dist_word / len_word) if len_word > 0 else 0.0

    #         # pred_ids = self.tokenizer.encode(pred_clean, add_special_tokens=False)
    #         # label_ids = self.tokenizer.encode(label_clean, add_special_tokens=False)
    #         pred_ids = uyghur_bpe.encode(pred_clean)
    #         label_ids = uyghur_bpe.encode(label_clean)



    #         dist_token = editdistance.eval(pred_ids, label_ids)
    #         len_token = max(len(pred_ids), len(label_ids))
    #         score_token = 1.0 - (dist_token / len_token) if len_token > 0 else 0.0

    #         raw_score = (w_word * score_word) + (w_char * score_char) + (w_token * score_token)
            
    #         shaped_score = raw_score ** sharpness

    #         final_reward = shaped_score

    #         if dist_char == 0:
    #             final_reward += 0.5
            
    #         rewards.append(torch.tensor(final_reward, dtype=torch.float32))
    #     # print("-----------------compute_custom_rewards---------------------------")
    #     # print(rewards)
    #     return rewards

    def compute_custom_rewards(self, messages: list[str], labels: list[str]) -> list[torch.Tensor]:
        """
        计算混合奖励：WER (词) + CER (字符) + SER (BBPE Token)
        """
        cer_weight = 1.0
        ser_weight = 1.0
        wer_weight = 1.0

        rewards = []
        for pred_text, label_text in zip(messages, labels):
            pred_clean = pred_text.replace('\n', '').strip().lower()
            label_clean = label_text.replace('\n', '').strip().lower()

            if not label_clean:
                rewards.append(torch.tensor(0.0 if pred_clean else 1.0))
                raise ValueError("not label_clean")
                continue

            clean_pred = pred_clean
            clean_gt = label_clean
            cer = calculate_cer(clean_gt, clean_pred)
            acc_char = 1.0 - cer


            pred_words = pred_clean.split()
            label_words = label_clean.split()
            if len(label_words) == 0:
                wer = 1.0
            else:
                # Levenshtein 库可以直接处理 list 列表计算距离
                wer_dist = Levenshtein.distance(pred_words, label_words)
                wer = wer_dist / len(label_words)
            
            wer = min(wer, 1.0)
            acc_word = 1.0 - wer

            pred_ids = self.tokenizer.encode(pred_clean, add_special_tokens=False)
            label_ids = self.tokenizer.encode(label_clean, add_special_tokens=False)
            dist_token = editdistance.eval(pred_ids, label_ids)
            len_token = max(len(pred_ids), len(label_ids))
            score_token = 1.0 - (dist_token / len_token) if len_token > 0 else 0.0




            accuracy_score = (cer_weight*acc_char + wer_weight*acc_word + score_token*ser_weight )*3.0

            len_penalty = 0.0
            len_pred = len(clean_pred)
            len_gt = len(clean_gt)

            if len_pred == 0:
                len_penalty = -10.0
            
            # 2. 长度过短惩罚 (防止模型偷懒只输出几个字)
            # 如果预测长度小于真值的 30%，给予重罚
            elif len_pred < len_gt * 0.3:
                len_penalty = -2.0
                
            # 3. 长度过长惩罚 (防止模型重复输出 loop)
            # 如果预测长度超过真值的 2倍
            elif len_pred > len_gt * 2.0:
                len_penalty = -2.0

            format_penalty = 0.0

            total_reward = accuracy_score + len_penalty + format_penalty

            rewards.append(torch.tensor(total_reward, dtype=torch.float32))
            
        return rewards

    @torch.no_grad()
    def get_rewards(
        self,
        queries: list["torch.Tensor"],
        responses: list["torch.Tensor"],
        labels = None
    ) -> list["torch.Tensor"]:
        r"""Compute scores using given reward model.

        Both inputs and outputs are put on CPU.
        """
        if self.finetuning_args.reward_model_type == "api":
            
            token_ids = [r.tolist() for r in responses]
            
            messages = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

            label_ids = labels['labels'].detach().clone()
            # print(token_ids)
            # print(label_ids)
            
            # 2. 将 -100 替换为 pad_token_id (如果没有 pad_token_id 则设为 0)
            # 这样做是为了让 decode 函数能正常运行，decode 出来的结果里这些位置会变成 <pad> 或者空
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            label_ids[label_ids == -100] = pad_token_id
            # print(label_ids)
            target_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            return self.compute_custom_rewards(messages, target_str)

        batch: dict[str, torch.Tensor] = self.prepare_model_inputs(queries, responses)
        unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="reward")
            reward_model = self.model
        else:
            reward_model = self.reward_model

        with unwrap_model_for_generation(reward_model, self.accelerator), self.amp_context:  # support bf16
            values: torch.Tensor = reward_model(**batch, return_dict=True, use_cache=False)[-1]

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="default")

        rewards = values.gather(dim=-1, index=(batch["attention_mask"].sum(dim=-1, keepdim=True) - 1))
        return rewards.float().detach()  # use fp32 type

    @override
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        r"""Calculate model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with self.amp_context:  # support bf16
                logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)
            # print("----------------------input_kwargs--------------------------")
            # print(input_kwargs['input_ids'])
            # print("----------------------query_batch--------------------------")
            # print(query_batch)
            # print("----------------------response_batch--------------------------")
            # print(response_batch)
            # print("----------------------all_logprobs--------------------------")
            # print(logits)
            # print(torch.argmax(logits, dim=-1))
            # preds = torch.argmax(logits, dim=-1)


            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]
            # # print("--------------------masks--------------------------")
            # # print(masks)
            # # print("------- 按 Batch 提取预测结果 -------")
            
            # mask_bool = masks.bool()
            # for i in range(len(preds)):
            #     # 取出第 i 个样本中，mask 为 True 的那些 token
            #     valid_tokens = preds[i][mask_bool[i]] 
                
            #     print(f"Batch {i} 对应的 TokenIDs: {valid_tokens}")

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @override
    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""Save model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            self._save(output_dir, state_dict=unwrapped_model.state_dict())
