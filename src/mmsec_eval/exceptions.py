# 中文注释：定义 MmsecError 的结构化职责，作为项目工程中状态、配置或行为的边界。
# 文件说明：该文件属于项目工程，集中实现 exceptions 相关逻辑。
class MmsecError(Exception):
    """Base exception for mmsec_eval."""


# 中文注释：定义 ConfigError 的结构化职责，作为项目工程中状态、配置或行为的边界。
class ConfigError(MmsecError):
    """Configuration is invalid."""


# 中文注释：定义 PluginNotFoundError 的结构化职责，作为项目工程中状态、配置或行为的边界。
class PluginNotFoundError(MmsecError):
    """Plugin does not exist in registry."""


# 中文注释：定义 ParseError 的结构化职责，作为项目工程中状态、配置或行为的边界。
class ParseError(MmsecError):
    """Document parsing failed."""

