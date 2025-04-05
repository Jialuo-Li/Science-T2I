from hydra.core.config_store import ConfigStore

from optimizers.adamw import AdamWOptimizerConfig
from optimizers.dummy_optimizer import DummyOptimizerConfig

cs = ConfigStore.instance()
cs.store(group="optimizer", name="dummy", node=DummyOptimizerConfig)
cs.store(group="optimizer", name="adamw", node=AdamWOptimizerConfig)
