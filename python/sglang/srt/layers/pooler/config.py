from sglang.srt.layers.pooler import PoolingType
from sglang.srt.server_args import ServerArgs


class PoolerConfig:
    def __init__(
        self, pooling_type: PoolingType | None = None, normalize: bool | None = None
    ):
        self.pooling_type = pooling_type

        # None is different from False because different models have different defaults
        # for unset 'normalize' config to maintain backward compatibility
        self.normalize = normalize

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        return PoolerConfig(
            pooling_type=(
                PoolingType[server_args.pooling_type]
                if server_args.pooling_type
                else None
            ),
        )

    def merge_with_defaults(
        self, pooling_type: PoolingType, normalize: bool
    ) -> "PoolerConfig":
        """Method to merge with model-specific defaults if the config(s) are not passed by the user"""

        self.pooling_type = self.pooling_type or pooling_type
        self.normalize = self.normalize if self.normalize is not None else normalize

        return self
