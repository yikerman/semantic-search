from functools import partial

from psycopg import sql

from semsearch.web.search.filters import (
    SqlPredicate,
    compile_filters,
    filter_by_language,
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
