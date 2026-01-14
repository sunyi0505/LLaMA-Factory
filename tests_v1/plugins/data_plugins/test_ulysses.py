import pytest
from transformers import AutoTokenizer
from llamafactory.v1.config.model_args import ModelArguments
from llamafactory.v1.config.data_args import DataArguments
from llamafactory.v1.plugins.data_plugins.sequence_parallel import SequenceParallelDataPlugin


@pytest.mark.parametrize("sequence_parallel_size", [2])
def test_sequence_parallel_data_plugin(sequence_parallel_size):
    examples = {'input_ids': [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]], 'position_ids': [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]],
                'labels': [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]], 'attention_mask': [[1]*15]}
    data_args = DataArguments()
    model_args = ModelArguments(model='Qwen/Qwen2.5-0.5B-Instruct', sequence_parallel_size=sequence_parallel_size, sequence_parallel_mode='ulysses')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    pad_data = SequenceParallelDataPlugin("pad_sequence")(examples, model_args, tokenizer, data_args)
    sp_data = SequenceParallelDataPlugin("sp_split")(pad_data, model_args, tokenizer, data_args)
    assert sp_data == {'input_ids': [[1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15, 151643]], 'position_ids': [[1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15,2047]], 'labels': [[2,3,4,5,6,7,8,9], [10,11,12,13,14,15,-100,-100]], 'attention_mask': [[1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,0]]}