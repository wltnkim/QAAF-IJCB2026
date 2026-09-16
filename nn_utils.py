"""Small training utilities."""
import torch


class MyDataParallel(torch.nn.DataParallel):
    """DataParallel that forwards unknown attribute lookups to the wrapped module.

    Plain DataParallel hides the wrapped model's own attributes and methods, so
    code written against the bare model breaks as soon as it is wrapped. This
    looks the attribute up on the wrapper first and falls back to the module.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
