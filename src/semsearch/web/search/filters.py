from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
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


def filter_by_published_range(
    published_from: date | None, published_to: date | None
) -> SearchFilter:
    return partial(
        _published_range_predicate,
        published_from=published_from,
        published_to=published_to,
    )


def _published_range_predicate(
    page_alias: str,
    *,
    published_from: date | None,
    published_to: date | None,
) -> SqlPredicate:
    column = sql.SQL("{}.published_at").format(sql.Identifier(page_alias))
    clauses: list[sql.Composable] = []
    params: list[object] = []
    if published_from is not None:
        clauses.append(sql.SQL("{} >= %s").format(column))
        params.append(datetime.combine(published_from, time.min, tzinfo=UTC))
    if published_to is not None:
        clauses.append(sql.SQL("{} < %s + INTERVAL '24 hours'").format(column))
        params.append(datetime.combine(published_to, time.min, tzinfo=UTC))
    return SqlPredicate(
        sql.SQL(" AND ").join(clauses) if clauses else sql.SQL("TRUE"),
        tuple(params),
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
