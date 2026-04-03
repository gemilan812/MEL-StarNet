"""
文件说明：
这个文件提供 Avg-CA only 版本的 MEL-StarNet 包装入口，
主要用于表 4.7 中 DCAFE 内部小消融实验。
"""

from 公共核心.mel_starnet_variants import starnet_s4_avg_ca

__all__ = ["starnet_s4_avg_ca"]
