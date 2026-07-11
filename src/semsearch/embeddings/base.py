from collections.abc import Awaitable, Callable


type EmbedDocuments = Callable[[list[str]], Awaitable[list[list[float]]]]
type EmbedQuery = Callable[[str], Awaitable[list[float]]]
