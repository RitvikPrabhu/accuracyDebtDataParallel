from core.registry import STRATEGIES
from .base import BaseStrategy

@STRATEGIES.register("tf_distribute")
class TFDistribute(BaseStrategy):
    name = "tf_distribute"
    def setup(self, model, optimizer, backend_cfg):
        # Placeholder – wire up tf.distribute if/when needed
        return model, optimizer
