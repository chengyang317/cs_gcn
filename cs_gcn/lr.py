


class WarmupScheduler(object):

    def __init__(self,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 **kwargs):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    '"{}" is not a supported type for warming up, valid types'
                    ' are "constant" and "linear"'.format(warmup))
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio

        self.base_lr = []  # initial lr for all param groups
        self._step = -1

    def _set_lr(self, optimizer, lr_groups):
        for param_group, lr in zip(optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def get_warmup_lr(self, cur_iters):
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.base_lr]
        elif self.warmup == 'linear':
            lr_scale = min(1., cur_iters / self.warmup_iters)
            k = (1 - lr_scale) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.base_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.base_lr]
        return warmup_lr

    def before_run(self, optimizer):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in optimizer.param_groups
        ]

    def step(self, optimizer, step=None):
        if step is None:
            cur_iter = self._step + 1
        else:
            cur_iter = step
        if cur_iter <= self.warmup_iters:
            warmup_lr = self.get_warmup_lr(cur_iter)
            self._set_lr(optimizer, warmup_lr)
