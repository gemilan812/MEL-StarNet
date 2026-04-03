"""
文件说明：
这个文件提供把 DS-CBS 改回 Original CBS 的模型入口，
主要用于表 4.6 的 DS-CBS 小消融实验。
"""

from 公共核心.mel_starnet_variants import starnet_s4_original_cbs

__all__ = ["starnet_s4_original_cbs"]
