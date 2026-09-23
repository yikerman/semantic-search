from collections.abc import Callable

from tokenizers import Tokenizer

# Allow room for provider-added EOS and document formatting tokens.
TOKEN_RESERVE = 32

type ValidateDocument = Callable[[str], None]


class DocumentTooLong(ValueError):
    pass


class TokenizerError(RuntimeError):
    pass


def load_tokenizer(identifier: str, revision: str) -> Tokenizer:
    try:
        tokenizer = Tokenizer.from_pretrained(identifier, revision=revision)
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return tokenizer
    except Exception as exc:
        raise TokenizerError(
            f"Could not load tokenizer {identifier} at revision {revision}"
        ) from exc


def validate_document(text: str, *, tokenizer: Tokenizer, max_tokens: int) -> None:
    count = len(tokenizer.encode(text, add_special_tokens=True).ids)
    budget = max_tokens - TOKEN_RESERVE
    if count > budget:
        raise DocumentTooLong(f"document has {count} tokens; input budget is {budget}")
