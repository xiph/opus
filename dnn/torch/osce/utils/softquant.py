import torch


class SoftQuant:
    name: str

    def __init__(self, names: str, scale: float) -> None:
        self.names = names
        self.quantization_noise = None
        self.scale = scale

    def __call__(self, module, inputs, *args, before=True):
        if not module.training: return

        if before:
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

        fn_before = lambda *x : fn(*x, before=True)
        fn_after = lambda *x : fn(*x, before=False)
        setattr(fn_before, 'sqm', fn)
        setattr(fn_after, 'sqm', fn)


        module.register_forward_pre_hook(fn_before)
        module.register_forward_hook(fn_after)

        module

        return fn


def soft_quant(module, names=['weight'], scale=0.5/127):
    fn = SoftQuant.apply(module, names, scale)
    return module

def remove_soft_quant(module, names=['weight']):
    for k, hook in module._forward_pre_hooks.items():
        if hasattr(hook, 'sqm'):
            if isinstance(hook.sqm, SoftQuant) and hook.sqm.names == names:
                del module._forward_pre_hooks[k]
    for k, hook in module._forward_hooks.items():
        if hasattr(hook, 'sqm'):
            if isinstance(hook.sqm, SoftQuant) and hook.sqm.names == names:
                del module._forward_hooks[k]

    return module