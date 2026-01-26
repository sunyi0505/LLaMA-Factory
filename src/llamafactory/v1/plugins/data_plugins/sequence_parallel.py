from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.v1.utils.plugin import BasePlugin
from llamafactory.v1.config.data_args import DataArguments
from llamafactory.v1.config.model_args import ModelArguments
from transformers import AutoTokenizer


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

@SequenceParallelDataPlugin('pad_sequence').register()
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
    examples["origin_attention_mask"] = examples["attention_mask"].copy()
    return examples

@SequenceParallelDataPlugin('sp_split').register()
def sp_split(examples, model_args, tokenizer, data_args=None):
    is_multimodal = hasattr(tokenizer, "image_token_id") and tokenizer.image_token_id is not None

    all_image_position_maps = list()
    new_examples = dict()

    for k, v in examples.items():
        chunks = list()
        for row in v:
            if row is None:
                chunks.extend([None] * model_args.sequence_parallel_size)
            elif k in ['images', 'origin_attention_mask']:
                chunks.extend([row] * model_args.sequence_parallel_size)
            else:
                chunks.extend(
                    preprocess_sp_dataset(row, model_args.sequence_parallel_size, model_args.sequence_parallel_mode)
                )
            if is_multimodal and k.endswith("input_ids") and len(all_image_position_maps) < len(v) * model_args.sequence_parallel_size:
                image_position_info = create_image_position_info(row, tokenizer.image_token_id)
                all_image_position_maps.extend(
                    preprocess_sp_dataset(image_position_info, model_args.sequence_parallel_size, model_args.sequence_parallel_mode)
                )
        new_examples[k] = chunks

    if is_multimodal and len(all_image_position_maps) > 0:
        new_examples["image_position_map"] = all_image_position_maps
        for index in range(len(new_examples["images"])):
            if all(image_position_info == -1 for image_position in new_examples["image_position_maps"][index]):
                new_examples["images"][index] = None

    return new_examples

if __name__ == "__main__":
    examples = {'input_ids': [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2,3,4,5,6,7,8]], 'position_ids': [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2,3,4,5,6,7,8]],
                'labels': [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2,3,4,5,6,7,8]], 'attention_mask': [[1]*15, [1]*8]}
    data_args = DataArguments()
    model_args = ModelArguments(model='xxx', sequence_parallel_size=2, sequence_parallel_mode='ulysses')
    tokenizer = AutoTokenizer.from_pretrained('xxx')
    pad_data = SequenceParallelDataPlugin("pad_sequence")(examples, model_args, tokenizer, data_args)
    print(SequenceParallelDataPlugin("sp_split")(pad_data, model_args, tokenizer, data_args))
