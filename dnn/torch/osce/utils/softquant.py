import torch


class SoftQuant:
    name: str

    def __init__(self, names: str, scale: float) -> None:
        self.names = names
        self.quantization_noise = None
        self.scale = scale

    def __call__(self, module, inputs, *args):
        if self.quantization_noise is None:
            self.quantization_noise = dict()
            for name in self.names:
                weight = getattr(module, name)
                self.quantization_noise[name] = \
                    self.scale * weight.abs().max() * 2 * (torch.rand_like(weight) - 0.5)
                with torch.no_grad():
                    weight.data[:] = weight + self.quantization_noise[name]
        else:
            for name in self.names:
                weight = getattr(module, name)
                with torch.no_grad():
                    weight.data[:] = weight - self.quantization_noise[name]
            self.quantization_noise = None

    def apply(module, names=['weight'], scale=0.5/127):
        fn = SoftQuant(names, scale)

        for name in names:
            if not hasattr(module, name):
                raise ValueError("")

        module.register_forward_pre_hook(fn)
        module.register_forward_hook(fn)

        module

        return fn


def soft_quant(module, names=['weight'], scale=1/127):
    fn = SoftQuant.apply(module, names, scale)
    return module

def remove_soft_quant(module, names=['weight']):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SoftQuant) and hook.names == names:
            del module._forward_pre_hooks[k]
    for k, hook in module._forward_hooks.items():
        if isinstance(hook, SoftQuant) and hook.names == names:
            del module._forward_hooks[k]

    return module