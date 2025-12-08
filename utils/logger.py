# utils/logger.py

import time


class TrainLogger:
    """
    非必须，但可以让日志更好看一点
    """

    def __init__(self):
        self.start_time = time.time()

    def log(self, epoch, iters, total_iters, losses):
        elapsed = time.time() - self.start_time
        msg = f"[Epoch {epoch}] [{iters}/{total_iters}] "
        loss_str = " ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
        msg += loss_str
        msg += f" | time: {elapsed:.1f}s"
        print(msg)
