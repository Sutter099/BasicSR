from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.MWUNet import MWUNet
from basicsr.archs.destripe_networks_util import MultiScaleDis

@ARCH_REGISTRY.register()
class MWUNetWrapper(MWUNet):
    def __init__(self, in_ch=3, out_ch=3):
        super(MWUNetWrapper, self).__init__(in_ch, out_ch)

@ARCH_REGISTRY.register()
class MultiScaleDisWrapper(MultiScaleDis):
    def __init__(self, input_dim=3, n_scale=3, n_layer=4, norm='None', sn=False):
        super(MultiScaleDisWrapper, self).__init__(input_dim, n_scale, n_layer, norm, sn)
