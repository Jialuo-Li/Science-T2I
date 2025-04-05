
from hydra.core.config_store import ConfigStore

from tasks.clip_task import CLIPTaskConfig

cs = ConfigStore.instance()
cs.store(group="task", name="clip", node=CLIPTaskConfig)

