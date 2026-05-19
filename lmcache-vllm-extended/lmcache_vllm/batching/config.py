from typing import Literal

from pydantic import BaseModel, ConfigDict

BatchStrategy = Literal["none", "grouped"]


class BatchSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    valid_strategies: tuple[BatchStrategy, ...] = ("none", "grouped")
    default_batch_size: int = 4
    default_batch_timeout_ms: int = 50
    context_separator: str = "###"
    batch_size_header: str = "x-lmcache-batch-size"
    batch_timeout_header: str = "x-lmcache-batch-timeout-ms"
    batch_strategy_header: str = "x-lmcache-scheduler"


batch_settings = BatchSettings()
