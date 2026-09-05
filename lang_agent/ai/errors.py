"""ai 层异常定义。"""


class AILayerError(Exception):
    """ai 层所有异常的基类。"""


class UnknownProviderError(AILayerError, ValueError):
    """请求的 provider 未注册。"""


class UnknownProtocolError(AILayerError, ValueError):
    """provider 不支持请求的 protocol。"""


class MissingApiKeyError(AILayerError, RuntimeError):
    """未找到 provider 所需的 API key。"""
