# 定义 `MmsecError` 的状态和行为边界，供项目工程在固定职责内复用。
# 文件说明：该文件属于项目工程，集中实现 exceptions 相关逻辑。
class MmsecError(Exception):
    """Base exception for mmsec_eval."""


# 定义 `ConfigError` 的状态和行为边界，供项目工程在固定职责内复用。
class ConfigError(MmsecError):
    """Configuration is invalid."""


# 定义 `PluginNotFoundError` 的状态和行为边界，供项目工程在固定职责内复用。
class PluginNotFoundError(MmsecError):
    """Plugin does not exist in registry."""


# 定义 `ParseError` 的状态和行为边界，供项目工程在固定职责内复用。
class ParseError(MmsecError):
    """Document parsing failed."""

