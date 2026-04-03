"""
文件说明：
这个文件提供去掉 Inv-Leaf/细化模块后的模型入口，
主要用于表 4.4 和表 4.8 中与特征细化相关的消融实验。
"""

from 公共核心.mel_starnet_variants import starnet_s4_wo_inv_leaf, starnet_s4_wo_refinement

__all__ = ["starnet_s4_wo_inv_leaf", "starnet_s4_wo_refinement"]
