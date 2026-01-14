import torch
import pytest
import torch.distributed as dist
from llamafactory.v1.config.model_args import ModelArguments
from llamafactory.v1.plugins.model_plugins.ulysses.sequence_parallel import SequenceParallelModelPlugin, SequenceParallelLossPlugin
from transformers import AutoModelForCausalLM

def compute_origin_loss():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = model.to(0)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 151643]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]])
    labels = torch.tensor([[1, 2, 3, 4, 5, 6, 7, -100]])
    position_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 2048]])
    inputs = {"input_ids": input_ids.to(0), "attention_mask": attention_mask.to(0), "labels": labels.to(0), "position_ids": position_ids.to(0)}
    return SequenceParallelLossPlugin('compute_loss')(model, inputs)

@pytest.mark.parametrize("sequence_parallel_size", [2])
def test_sequence_parallel_loss_plugin(sequence_parallel_size):
    model_args = ModelArguments(model="Qwen/Qwen2.5-0.5B-Instruct", sequence_parallel_size=sequence_parallel_size, sequence_parallel_mode='ulysses')
    dist.init_process_group(backend="hccl", world_size=2, rank=-1)
    sp_group = SequenceParallelModelPlugin('apply_sequence_parallel')(model_args)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    rank = torch.distributed.get_rank()
    model = model.to(rank)

    model.sequence_parallel_group = sp_group
    if rank == 0:
        input_ids = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.tensor([[1, 1, 1, 1]])
        labels = torch.tensor([[2, 3, 4, 5]])
        position_ids = torch.tensor([[1, 2, 3, 4]])
    else:
        input_ids = torch.tensor([[5, 6, 7, 151643]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])
        labels = torch.tensor([[6, 7, -100, -100]])
        position_ids = torch.tensor([[5, 6, 7, 2048]])
    inputs = {"input_ids": input_ids.to(rank), "attention_mask": attention_mask.to(rank), "labels": labels.to(rank), "position_ids": position_ids.to(rank)}
    assert SequenceParallelLossPlugin('sequence_parallel_loss')(model, inputs) == compute_origin_loss()