from hydra.core.config_store import ConfigStore

from models.clip_model import ClipModelConfig

cs = ConfigStore.instance()
cs.store(group="model", name="clip", node=ClipModelConfig)

