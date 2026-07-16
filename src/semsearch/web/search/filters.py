from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

from psycopg import sql


@dataclass(frozen=True, slots=True)
class SqlPredicate:
    clause: sql.Composable
    params: tuple[object, ...] = ()


type SearchFilter = Callable[[str], SqlPredicate]


def filter_by_language(language: str) -> SearchFilter:
    return partial(_language_predicate, language=language)


def _language_predicate(page_alias: str, *, language: str) -> SqlPredicate:
    return SqlPredicate(
        sql.SQL("{}.language = %s").format(sql.Identifier(page_alias)),
        (language,),
    )


def compile_filters(
    filters: Sequence[SearchFilter], *, page_alias: str
) -> SqlPredicate:
    predicates = [item(page_alias) for item in filters]
    if not predicates:
        return SqlPredicate(sql.SQL("TRUE"))
    return SqlPredicate(
        sql.SQL(" AND ").join(predicate.clause for predicate in predicates),
        tuple(param for predicate in predicates for param in predicate.params),
    )
