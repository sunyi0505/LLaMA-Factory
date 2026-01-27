# Copyright 2025 HuggingFace Inc., the KVCache.AI team, Approaching AI, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

import os
import random
from datasets import load_dataset
from typing import TYPE_CHECKING, Any

import torch
from transformers import BitsAndBytesConfig, EetqConfig, GPTQConfig, HqqConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ...utils import logging
from ...utils.constants import QuantizationMethod, FILEEXT2TYPE
from ...accelerator.helper import get_current_device
from ...utils.packages import check_version
from ...utils.plugin import BasePlugin
from ...config.model_args import ModelArguments

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer

logger = logging.get_logger(__name__)


class QuantizationPlugin(BasePlugin):
    r"""Plugin for model quantization."""

    def __call__(
        self,
        init_kwargs: dict[str, Any]=None,
        config: "PretrainedConfig"=None,
        tokenizer: "PreTrainedTokenizer"=None,
        model_args: "ModelArguments"=None,
        is_trainable: bool=False,
    ) -> None:
        return super().__call__(init_kwargs, config=config, tokenizer=tokenizer, model_args=model_args, is_trainable=is_trainable)


def _get_quantization_dataset(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> list[dict[str, Any]]:
    r"""Prepare the tokenized dataset to perform AutoGPTQ. Do not use tensor output for JSON serialization."""
    if os.path.isfile(model_args.quant_config.get("export_quantization_dataset")):
        data_path = FILEEXT2TYPE.get(model_args.quant_config.get("export_quantization_dataset").split(".")[-1], None)
        data_files = model_args.quant_config.get("export_quantization_dataset")
    else:
        data_path = model_args.quant_config.get("export_quantization_dataset")
        data_files = None

    dataset = load_dataset(
        path=data_path,
        data_files=data_files,
        split="train",
        cache_dir=model_args.quant_config.get("cache_dir"),
        token=model_args.quant_config.get("hf_hub_token"),
    )

    samples = []
    maxlen = model_args.quant_config.get("export_quantization_maxlen")
    for _ in range(model_args.quant_config.get("export_quantization_nsamples")):
        n_try = 0
        while True:
            if n_try > 100:
                raise ValueError("Cannot find satisfying example, considering decrease `export_quantization_maxlen`.")

            sample_idx = random.randint(0, len(dataset) - 1)
            sample: dict[str, torch.Tensor] = tokenizer(dataset[sample_idx]["text"], return_tensors="pt")
            n_try += 1
            if sample["input_ids"].size(1) > maxlen:
                break  # TODO: fix large maxlen

        word_idx = random.randint(0, sample["input_ids"].size(1) - maxlen - 1)
        input_ids = sample["input_ids"][:, word_idx : word_idx + maxlen]
        attention_mask = sample["attention_mask"][:, word_idx : word_idx + maxlen]
        samples.append({"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist()})

    return samples

def _check_quantization_method(quantization_config) -> None:
    r"""Get quantization bit from model args."""
    quant_method = quantization_config.get("quant_method", "")

    if quant_method not in (QuantizationMethod.MXFP4, QuantizationMethod.FP8) and (
        is_deepspeed_zero3_enabled() or is_fsdp_enabled()
    ):
        raise ValueError("DeepSpeed ZeRO-3 or FSDP is incompatible with PTQ-quantized models.")
    quant_bits = quantization_config.get("bits", "?")
    logger.info_rank0(f"Loading {quant_bits}-bit {quant_method.upper()}-quantized model.")

@QuantizationPlugin("mxfp4").register()
def quantization_with_mxfp4(
    init_kwargs: dict[str, Any],
    config: "PretrainedConfig"=None,
    **kwargs,
):
    r"""Quantization with MXFP4."""
    quantization_config: dict[str, Any] = getattr(config, "quantization_config", None)
    assert quantization_config is not None, "MXFP4 quantization requires `quantization_config` in model config."
    from transformers import Mxfp4Config

    quant_config = Mxfp4Config(dequantize=True)
    init_kwargs["quantization_config"] = quant_config
    init_kwargs["ignore_mismatched_sizes"] = True
    _check_quantization_method(quantization_config)
    return init_kwargs

@QuantizationPlugin("fp8").register()
def quantization_with_fp8(
    init_kwargs: dict[str, Any],
    config: "PretrainedConfig"=None,
    **kwargs,
) -> None:
    r"""Quantization with FineGrainedFP8."""
    quantization_config: dict[str, Any] = getattr(config, "quantization_config", None)
    assert quantization_config is not None, "FP8 quantization requires `quantization_config` in model config."
    from transformers import FineGrainedFP8Config

    quant_config = FineGrainedFP8Config(dequantize=True)
    init_kwargs["quantization_config"] = quant_config
    init_kwargs["ignore_mismatched_sizes"] = True
    _check_quantization_method(quantization_config)
    return init_kwargs

@QuantizationPlugin("gptq").register()
def quantization_with_gptq(
    init_kwargs: dict[str, Any],
    config: "PretrainedConfig"=None,
    **kwargs,
) -> None:
    r"""Quantization with GPTQ."""
    quantization_config: dict[str, Any] = getattr(config, "quantization_config", None)
    assert quantization_config is not None, "GPTQ quantization requires `quantization_config` in model config."
    check_version("gptqmodel>=2.0.0", mandatory=True)

    quantization_config.pop("disable_exllama", None)  # remove deprecated args
    quantization_config["use_exllama"] = False  # disable exllama
    _check_quantization_method(quantization_config)
    return init_kwargs

@QuantizationPlugin("awq").register()
def quantization_with_awq(
    init_kwargs: dict[str, Any],
    config: "PretrainedConfig"=None,
    **kwargs,
) -> None:
    r"""Quantization with AWQ."""
    quantization_config: dict[str, Any] = getattr(config, "quantization_config", None)
    assert quantization_config is not None, "AWQ quantization requires `quantization_config` in model config."
    check_version("autoawq", mandatory=True)
    _check_quantization_method(quantization_config)
    return init_kwargs

@QuantizationPlugin("aqlm").register()
def quantization_with_aqlm(
    init_kwargs: dict[str, Any],
    config: "PretrainedConfig"=None,
    **kwargs,
) -> None:
    r"""Quantization with AQLM."""
    quantization_config: dict[str, Any] = getattr(config, "quantization_config", None)
    assert quantization_config is not None, "AQLM quantization requires `quantization_config` in model config."
    check_version("aqlm>=1.1.0", mandatory=True)
    quantization_config["bits"] = 2
    _check_quantization_method(quantization_config)
    return init_kwargs

@QuantizationPlugin("gptmodel").register()
def quantization_with_gptmodel(
    init_kwargs: dict[str, Any],
    config: "PretrainedConfig"=None,
    model_args: "ModelArguments"=None,
    tokenizer: "PreTrainedTokenizer"=None,
    **kwargs,
) -> None:
    r"""Quantization with GPTQModel."""
    assert "export_quantization_bit" in model_args.quant_config, "Please specify `export_quantization_bit` for AutoGPTQ quantization."
    assert model_args.quant_config.get("export_quantization_bit") in [8, 4, 3, 2], "AutoGPTQ only accepts 2/3/4/8-bit quantization."

    check_version("optimum>=1.24.0", mandatory=True)
    check_version("gptqmodel>=2.0.0", mandatory=True)
    from accelerate.utils import get_max_memory

    if getattr(config, "model_type", None) == "chatglm":
        raise ValueError("ChatGLM model is not supported yet.")

    try:
        from optimum.gptq import utils as gq_utils

        if "language_model.model.layers" not in gq_utils.BLOCK_PATTERNS:
            gq_utils.BLOCK_PATTERNS.insert(0, "language_model.model.layers")
    except ImportError:
        pass

    block_name_to_quantize = None
    if getattr(config, "model_type", None) in ["gemma3", "paligemma"]:
        block_name_to_quantize = "language_model.model.layers"

    init_kwargs["quantization_config"] = GPTQConfig(
        bits=model_args.quant_config.get("export_quantization_bit"),
        tokenizer=tokenizer,
        dataset=_get_quantization_dataset(tokenizer, model_args),
        block_name_to_quantize=block_name_to_quantize,
    )
    init_kwargs["device_map"] = "auto"
    init_kwargs["max_memory"] = get_max_memory()
    model_args.quant_config["compute_dtype"] = torch.float16  # force fp16 for gptqmodel
    logger.info_rank0(f"Quantizing model to {model_args.quant_config.get('export_quantization_bit')} bit with GPTQModel.")
    return init_kwargs

@QuantizationPlugin("bnb").register()
def quantization_with_bnb(
    init_kwargs: dict[str, Any],
    model_args: "ModelArguments"=None,
    **kwargs,
) -> None:
    r"""Quantization with BNB."""
    assert "quantization_bit" in model_args.quant_config, "Please specify `quantization_bit` for Bitsandbytes quantization."
    assert model_args.quant_config.get("quantization_bit") in [8, 4], "Bitsandbytes only accepts 4-bit or 8-bit quantization."
    if model_args.quant_config.get("quantization_bit") == 8:
        check_version("bitsandbytes>=0.37.0", mandatory=True)
        init_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif model_args.quant_config.get("quantization_bit") == 4:
        check_version("bitsandbytes>=0.39.0", mandatory=True)
        init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_args.quant_config.get("compute_dtype", torch.float16),
            bnb_4bit_use_double_quant=model_args.quant_config.get("double_quantization", True),
            bnb_4bit_quant_type=model_args.quant_config.get("quantization_type", "nf4"),
            bnb_4bit_quant_storage=model_args.quant_config.get("compute_dtype", torch.float16),  # crucial for fsdp+qlora
        )
    else:
        raise ValueError("Bitsandbytes only accepts 4-bit or 8-bit quantization.")

    # Do not assign device map if:
    # 1. deepspeed zero3 or fsdp (train)
    # 2. auto quantization device map (inference)
    if is_deepspeed_zero3_enabled() or is_fsdp_enabled() or model_args.quant_config.get("quantization_device_map") == "auto":
        if model_args.quant_config.get("quantization_bit") != 4:
            raise ValueError("Only 4-bit quantized model can use fsdp+qlora or auto device map.")

        check_version("bitsandbytes>=0.43.0", mandatory=True)
    else:
        init_kwargs["device_map"] = {"": get_current_device()}  # change auto device map for inference

    logger.info_rank0(f"Quantizing model to {model_args.quant_config.get('quantization_bit')} bit with bitsandbytes.")
    return init_kwargs

@QuantizationPlugin("hqq").register()
def quantization_with_hqq(
    init_kwargs: dict[str, Any],
    model_args: "ModelArguments"=None,
    **kwargs,
) -> None:
    r"""Quantization with HQQ."""
    assert "quantization_bit" in model_args.quant_config, "Please specify `quantization_bit` for HQQ quantization."
    assert model_args.quant_config.get("quantization_bit") in [8, 6, 5, 4, 3, 2, 1], "HQQ only accepts 1/2/3/4/5/6/8-bit quantization."

    if is_deepspeed_zero3_enabled() or is_fsdp_enabled():
        raise ValueError("HQQ quantization is incompatible with DeepSpeed ZeRO-3 or FSDP.")

    check_version("hqq", mandatory=True)
    init_kwargs["quantization_config"] = HqqConfig(
        nbits=model_args.quant_config.get("quantization_bit"), quant_zero=False, quant_scale=False, axis=0
    )  # use ATEN kernel (axis=0) for performance
    logger.info_rank0(f"Quantizing model to {model_args.quant_config.get('quantization_bit')} bit with HQQ.")
    return init_kwargs

@QuantizationPlugin("eetq").register()
def quantization_with_eetq(
    init_kwargs: dict[str, Any],
    model_args: "ModelArguments"=None,
    **kwargs,
) -> None:
    r"""Quantization with EETQ."""
    assert "quantization_bit" in model_args.quant_config, "Please specify `quantization_bit` for EETQ quantization."
    assert model_args.quant_config.get("quantization_bit") == 8, "EETQ only accepts 8-bit quantization."

    if is_deepspeed_zero3_enabled() or is_fsdp_enabled():
        raise ValueError("EETQ quantization is incompatible with DeepSpeed ZeRO-3 or FSDP.")

    check_version("eetq", mandatory=True)
    init_kwargs["quantization_config"] = EetqConfig()
    logger.info_rank0(f"Quantizing model to {model_args.quant_config.get('quantization_bit')} bit with EETQ.")
    return init_kwargs
