from torch.optim.lr_scheduler import _LRScheduler

class LR_Scheduler(_LRScheduler):
    def __init__(self, optimizer, config):
        self.config = config
        self.initial_lr = self.config.TRAIN.LR_Scheduler.initial_lr
        self.final_lr = self.config.TRAIN.LR_Scheduler.final_lr
        self.decay_factor = self.config.TRAIN.LR_Scheduler.decay_factor
        self.current_lr = self.initial_lr
        self.optimizer = optimizer
        super(LR_Scheduler, self).__init__(optimizer)

    def get_lr(self):
        return [self.current_lr] * len(self.optimizer.param_groups)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Decay the learning rate by 0.5 every 100 epochs
        if epoch % 20 == 0:
            if self.current_lr > self.final_lr:
                self.current_lr *= self.decay_factor
            else:
                self.current_lr = self.final_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr