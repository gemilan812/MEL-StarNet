"""
文件说明：
这个文件提供用 CBAM 替换 DCAFE 的模型入口，
主要用于表 4.5 注意力模块替换实验。
"""

from 公共核心.mel_starnet_variants import starnet_s4_cbam

__all__ = ["starnet_s4_cbam"]
