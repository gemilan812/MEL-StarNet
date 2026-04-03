"""
文件说明：
这个文件提供去掉 DCAFE 后的模型入口，
主要用于表 4.4 的总体模块去除式消融实验。
"""

from 公共核心.mel_starnet_variants import starnet_s4_wo_dcafe

__all__ = ["starnet_s4_wo_dcafe"]
