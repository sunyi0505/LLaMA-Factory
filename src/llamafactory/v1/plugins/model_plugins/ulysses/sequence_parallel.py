import sys
from functools import partial
import torch
from llamafactory.v1.utils.plugin import BasePlugin
from llamafactory.v1.config.model_args import ModelArguments
import torch.distributed as dist
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from llamafactory.v1.plugins.model_plugins.ulysses.ulysses import UlyssesAttention
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from llamafactory.v1.config.data_args import DataArguments
from llamafactory.v1.plugins.data_plugins.sequence_parallel import SequenceParallelDataPlugin
from accelerate.accelerator import Accelerator
from accelerate.utils import DistributedType


class SequenceParallelModelPlugin(BasePlugin):
    def __call__(self, model_args, full_determinism=True):
        return super().__call__(model_args, full_determinism=full_determinism)
    

class SequenceParallelLossPlugin(BasePlugin):
    def __call__(self, model, inputs, *args, **kwargs):
        return super().__call__(model, inputs, *args, **kwargs)

def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    sequence_parallel_size=1,
    dropout=0,
    deterministic=False,
    is_causal=True,
    group=None,
    mode="ulysses",
    attn_fn=None,
    target_dtype=None,
    **kwargs,
):
    if mode == "ulysses":
        dist_attn = UlyssesAttention(sequence_process_group=group, attn_fn=attn_fn)
        attn_output = dist_attn(query_states, key_states, value_states, kwargs['origin_attention_mask'],
                                query_length=query_states.shape[1] * sequence_parallel_size, deterministic=deterministic, dropout_p=dropout, causal=is_causal,
                                position_ids=kwargs.get("position_ids", None), target_dtype=target_dtype)
    else:
        raise NotImplementedError('Other sequence parallel modes are to be implemented.')

    return attn_output


def init_sp_group(sp_size):
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    assert world_size % sp_size == 0, "Total number of GPUs must be a multiple of sequence_parallel_size."

    sp_group_num = world_size // sp_size
    sp_ranks_list = [list(range(i * sp_size, i * sp_size + sp_size)) for i in range(sp_group_num)]

    sp_groups = [dist.new_group(sp_ranks_this) for sp_ranks_this in sp_ranks_list]

    global_rank_this = dist.get_rank()
    sp_idx = global_rank_this // sp_size
    return sp_groups[sp_idx]

def backward(self, loss, **kwargs):
    learning_rate = kwargs.get("learning_rate")

    if self.distributed_type != DistributedType.DEEPSPEED:
        loss = loss / self.gradient_accumulation_steps
    if self.distributed_type == DistributedType.DEEPSPEED:
        self.deepspeed_engine_wrapper.backward(loss, sync_gradients=not self.sync_gradients, **kwargs)
    elif self.distributed_type == DistributedType.MEGATRON_LM:
        return
    elif self.scaler is not None:
        self.scaler.scale(loss).backward(**kwargs)
    elif learning_rate is not None and self.has_lomo_optimizer:
        self.lomo_backward(loss, learning_rate)
    else:
        loss.backward(**kwargs)
    for _, param in self._models[0].named_parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= dist.get_world_size()

@SequenceParallelModelPlugin('apply_sequence_parallel').register()
def apply_sequence_parallel(model_args, full_determinism=True):
    if model_args.sequence_parallel_size == 1:
        return None  # no sequence parallelism
    
    # init sequence-parallel groups here
    group_this = init_sp_group(model_args.sequence_parallel_size)
    original_attn = transformers.modeling_flash_attention_utils._flash_attention_forward

    try:
        # old_flash_attention_forward = transformers.modeling_flash_attention_utils._flash_attention_forward
        if model_args.sequence_parallel_mode == 'ulysses':
            new_flash_attention_forward = partial(new_flash_attn_forward, group=group_this, mode=model_args.sequence_parallel_mode, deterministic=full_determinism, attn_fn=original_attn, sequence_parallel_size=model_args.sequence_parallel_size)
            # assert check_params(old_flash_attention_forward, new_flash_attention_forward)
        else:
            raise NotImplementedError('Other sequence parallel modes are to be implemented.')
        
        # monkey patching
        transformers.modeling_flash_attention_utils._flash_attention_forward = new_flash_attention_forward
        Accelerator.backward = backward

        for module_name, module in list(sys.modules.items()):
            try:
                if (hasattr(module, '__file__') and 'transformers' in module.__file__ and getattr(module._flash_attention_forward, '__name__', '') == '_flash_attention_forward'):
                    module._flash_attention_forward = new_flash_attention_forward
                    print(f"Patched _flash_attention_forward in module: {module_name}")
            except (AttributeError, TypeError):
                continue

    except:
        raise ValueError(
            f"The current transformer version {transformers.__version__} is not supported. "
        )

    return group_this

@SequenceParallelLossPlugin('sequence_parallel_loss').register()
def sequence_parallel_loss(model, inputs, **kwargs):
    loss_fct = CrossEntropyLoss(reduction="sum")

    outputs = model(**inputs)

    logits, labels = outputs["logits"] if isinstance(outputs, dict) else outputs[1], inputs["labels"]

    # shift labels
    world_size = dist.get_world_size(model.sequence_parallel_group)
    rank = dist.get_rank(model.sequence_parallel_group)
    global_labels = [torch.zeros_like(labels) for _ in range(world_size)]
    dist.all_gather(global_labels, labels, group=model.sequence_parallel_group)
    global_labels = torch.cat(global_labels, dim=1).contiguous()
    global_labels = nn.functional.pad(global_labels, (0, 1), value=loss_fct.ignore_index)
    shifted_labels = global_labels[..., 1:].contiguous()
    labels = torch.chunk(shifted_labels, world_size, dim=1)[rank].contiguous()

    logits = logits.float()

    vocab_size = model.config.vocab_size
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)

    labels = labels.to(logits.device)
    loss = loss_fct(logits, labels)

    sp_group = model.sequence_parallel_group
    loss = dist.nn.all_reduce(loss, op=dist.ReduceOp.SUM, group=sp_group)

    label_num = (labels != loss_fct.ignore_index).sum()
    label_num = dist.nn.all_reduce(label_num, op=dist.ReduceOp.SUM, group=sp_group)

    loss /= label_num
    return loss

@SequenceParallelLossPlugin('compute_loss').register()
def compute_loss(model, inputs, **kwargs):
    loss_fct = CrossEntropyLoss(reduction="sum")

    outputs = model(**inputs)

    logits, labels = outputs["logits"] if isinstance(outputs, dict) else outputs[1], inputs["labels"]

    logits = logits.float()

    labels = nn.functional.pad(labels, (0, 1), value=loss_fct.ignore_index)
    labels = labels[..., 1:].contiguous()

    vocab_size = model.config.vocab_size
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)

    labels = labels.to(logits.device)
    loss = loss_fct(logits, labels)

    label_num = (labels != loss_fct.ignore_index).sum()

    loss /= label_num
    return loss

if __name__ == "__main__": # for testing torchrun --nproc_per_node=2 sequence_parallel.py
    data_args = DataArguments()
    model_args = ModelArguments(model="xxx", sequence_parallel_size=2, sequence_parallel_mode='ulysses')
    torch.distributed.init_process_group(backend="hccl", world_size=2, rank=-1)
    if model_args.sequence_parallel_size > 1:
        sp_group = SequenceParallelModelPlugin('apply_sequence_parallel')(model_args)
    model = AutoModelForCausalLM.from_pretrained(model_args.model, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model)
    rank = torch.distributed.get_rank()
    model = model.to(rank)

    input_ids = [[1, 2, 3, 4, 5, 6, 7, 151643]]
    attention_mask = [[1, 1, 1, 1, 1, 1, 1, 0]]
    labels = [[1, 2, 3, 4, 5, 6, 7, -100]]
    position_ids = [[1, 2, 3, 4, 5, 6, 7, 2048]]

    if model_args.sequence_parallel_size == 1:
        input_ids = torch.tensor(input_ids, device=f"npu:{rank}")
        attention_mask = torch.tensor(attention_mask, device=f"npu:{rank}")
        labels = torch.tensor(labels, device=f"npu:{rank}")
        position_ids = torch.tensor(position_ids, device=f"npu:{rank}")
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "position_ids": position_ids}
        print(SequenceParallelLossPlugin('compute_loss')(model, inputs))
    else:
        model.sequence_parallel_group = sp_group
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "position_ids": position_ids}
        pad_data = SequenceParallelDataPlugin("pad_sequence")(inputs, model_args, tokenizer, data_args)
        sp_data = SequenceParallelDataPlugin("sp_split")(pad_data, model_args, tokenizer, data_args)
        input_ids = torch.tensor([sp_data["input_ids"][rank]], device=f"npu:{rank}")
        origin_attention_mask = torch.tensor(attention_mask, device=f"npu:{rank}")
        attention_mask = torch.tensor([sp_data["attention_mask"][rank]], device=f"npu:{rank}")
        labels = torch.tensor([sp_data["labels"][rank]], device=f"npu:{rank}")
        position_ids = torch.tensor([sp_data["position_ids"][rank]], device=f"npu:{rank}")
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "position_ids": position_ids, "origin_attention_mask": origin_attention_mask}
        print(SequenceParallelLossPlugin('sequence_parallel_loss')(model, inputs))
