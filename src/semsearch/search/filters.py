from collections.abc import Callable, Sequence
from dataclasses import dataclass

from psycopg import sql


@dataclass(frozen=True, slots=True)
class SqlPredicate:
    clause: sql.Composable
    params: tuple[object, ...] = ()


type SearchFilter = Callable[[str], SqlPredicate]


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
