"""
文件说明：
这个文件提供统一的模型注册表和 get_model 接口，
便于按名字创建论文实验中用到的各种基线模型和消融模型。
"""

from typing import Callable, Dict

import torch.nn as nn

from 公共核心.mel_starnet_variants import CUSTOM_MODEL_FACTORY
from 表4_2_实验结果对比表.benchmark_models import BENCHMARK_MODEL_FACTORY


MODEL_ZOO: Dict[str, Callable[..., nn.Module]] = {}
MODEL_ZOO.update(BENCHMARK_MODEL_FACTORY)
MODEL_ZOO.update(CUSTOM_MODEL_FACTORY)


def get_model(model_name: str, **kwargs) -> nn.Module:
    if model_name not in MODEL_ZOO:
        available = ", ".join(sorted(MODEL_ZOO.keys()))
        raise KeyError(f"Unsupported model: {model_name}. Available models: {available}")
    return MODEL_ZOO[model_name](**kwargs)


__all__ = ["MODEL_ZOO", "get_model"]
