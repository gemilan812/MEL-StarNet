"""
文件说明：
这个文件提供仅保留 Involution、不加残差融合的模型入口，
主要用于表 4.8 的 Inv-Leaf 小消融实验。
"""

from 公共核心.mel_starnet_variants import starnet_s4_involution_only

__all__ = ["starnet_s4_involution_only"]
