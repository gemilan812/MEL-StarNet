"""
文件说明：
这个文件提供“+ DS-CBS + LED-CG + DCAFE”版本入口，
用于表 4.3 中接近完整模型但还未加入 Inv-Leaf 的累加消融实验。
"""

from 公共核心.mel_starnet_variants import starnet_s4_ds_led_cg_dcafe

__all__ = ["starnet_s4_ds_led_cg_dcafe"]
