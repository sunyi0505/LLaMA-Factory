from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.v1.utils.plugin import BasePlugin


class SequenceParallelDataPlugin(BasePlugin):
    def __call__(self, examples, model_args, tokenizer, data_args=None):
        return super().__call__(examples, model_args, tokenizer, data_args=data_args)

# modified from https://github.com/jzhang38/EasyContext/
def preprocess_sp_dataset(seq_ids, world_size, sequence_parallel_mode):
    if sequence_parallel_mode == 'ulysses':
        step = len(seq_ids) // world_size
        local_values = [seq_ids[s : s + step] for s in range(0, len(seq_ids), step)]
        return local_values
    else:
        raise NotImplementedError('Other sequence parallel modes are to be implemented.')

def get_max_lengths(examples):
    valid_lists = []
    for key, value in examples.items():
        if key.endswith("input_ids") and value is not None:
            valid_lists.append(value)
    if not valid_lists:
        return []

    max_lengths = [max(len(lst) if lst is not None else 0 for lst in group) for group in zip(*valid_lists)]

    return max_lengths

def create_image_position_info(seq_ids, image_token_id):
    info = []
    global_image_pos = 0

    for token_id in seq_ids:
        if token_id == image_token_id:
            info.append(global_image_pos)
            global_image_pos += 1
        else:
            info.append(-1)
    return info

@SequenceParallelDataPlugin('pad_sequence').register
def pad_sequence(examples, model_args, tokenizer, data_args):
    max_lengths = data_args.cutoff_len
    input_pad_token_id = tokenizer.pad_token_id
    assert data_args.ignore_pad_token_for_loss
    label_pad_token_id = IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    max_input_ids_length_list = get_max_lengths(examples)
    for k, v in examples.items():
        if k.endswith("input_ids"):
            pad_token_id = input_pad_token_id
        elif k.endswith("labels"):
            pad_token_id = label_pad_token_id
        elif k.endswith("attention_mask"):
            pad_token_id = 0
        elif k.endswith("position_ids"):
            pad_token_id = max_lengths - 1
        elif k == "images" or k == "videos" or k == "audios":
            # do not pad multimodal inputs here
            continue
        else:
            raise ValueError(f"Unsupported key for padding: {k}")
        
        for i in range(len(v)):
            tmp_sp_len = max_input_ids_length_list[i] // model_args.sequence_parallel_size
            closest_cutoff_len = int((tmp_sp_len + (8 - tmp_sp_len % 8)) * model_args.sequence_parallel_size)
            max_length = min(closest_cutoff_len, data_args.cutoff_len)

            v[i].extend([pad_token_id] * (max_length - len(v[i])))
            
        examples[k] = v

    return examples

@SequenceParallelDataPlugin('sp_split').register
def sp_split(examples, model_args, tokenizer, data_args=None):

    new_examples = dict()

    for k, v in examples.items():
        chunks = list()
        for row in v:
            if row is None:
                chunks.extend([None] * model_args.sequence_parallel_size)
            elif k in ['images']:
                chunks.extend([row] * model_args.sequence_parallel_size)
            else:
                chunks.extend(
                    preprocess_sp_dataset(row, model_args.sequence_parallel_size, model_args.sequence_parallel_mode)
                )
        new_examples[k] = chunks

    return new_examples
