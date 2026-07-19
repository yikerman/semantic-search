from datetime import UTC, date, datetime
from functools import partial

from psycopg import sql

from semsearch.web.search.filters import (
    SqlPredicate,
    compile_filters,
    filter_by_language,
    filter_by_published_range,
)


def column_equals(page_alias: str, *, column: str, value: object) -> SqlPredicate:
    return SqlPredicate(
        sql.SQL("{}.{} = %s").format(
            sql.Identifier(page_alias), sql.Identifier(column)
        ),
        (value,),
    )


def test_no_filters_compile_to_true():
    predicate = compile_filters([], page_alias="p")

    assert predicate.clause.as_string() == "TRUE"
    assert predicate.params == ()


def test_filters_combine_with_and_and_preserve_parameter_order():
    predicate = compile_filters(
        [
            partial(column_equals, column="site_id", value=3),
            partial(column_equals, column="published_at", value="2025-01-01"),
        ],
        page_alias="p",
    )

    assert (
        predicate.clause.as_string() == '"p"."site_id" = %s AND "p"."published_at" = %s'
    )
    assert predicate.params == (3, "2025-01-01")


def test_language_filter_targets_the_page_alias():
    predicate = compile_filters([filter_by_language("fr")], page_alias="page")

    assert predicate.clause.as_string() == '"page".language = %s'
    assert predicate.params == ("fr",)


def test_published_range_filter_uses_inclusive_utc_calendar_dates():
    predicate = filter_by_published_range(date(2025, 1, 2), date(2025, 3, 4))("page")

    assert predicate.clause.as_string() == (
        '"page".published_at >= %s AND "page".published_at < %s + INTERVAL \'24 hours\''
    )
    assert predicate.params == (
        datetime(2025, 1, 2, tzinfo=UTC),
        datetime(2025, 3, 4, tzinfo=UTC),
    )


def test_published_range_filter_supports_either_endpoint():
    lower = filter_by_published_range(date(2025, 1, 2), None)("p")
    upper = filter_by_published_range(None, date(2025, 3, 4))("p")
    empty = filter_by_published_range(None, None)("p")

    assert lower.clause.as_string() == '"p".published_at >= %s'
    assert lower.params == (datetime(2025, 1, 2, tzinfo=UTC),)
    assert upper.clause.as_string() == ("\"p\".published_at < %s + INTERVAL '24 hours'")
    assert upper.params == (datetime(2025, 3, 4, tzinfo=UTC),)
    assert empty.clause.as_string() == "TRUE"
    assert empty.params == ()
