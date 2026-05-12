class MmsecError(Exception):
    """Base exception for mmsec_eval."""


class ConfigError(MmsecError):
    """Configuration is invalid."""


class PluginNotFoundError(MmsecError):
    """Plugin does not exist in registry."""


class ParseError(MmsecError):
    """Document parsing failed."""

