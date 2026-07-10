from dataclasses import dataclass

from psycopg import sql

from semsearch.search.filters import SqlPredicate, compile_filters


@dataclass(frozen=True)
class ExampleFilter:
    column: str
    value: object

    def compile(self, page_alias: str) -> SqlPredicate:
        return SqlPredicate(
            sql.SQL("{}.{} = %s").format(
                sql.Identifier(page_alias), sql.Identifier(self.column)
            ),
            (self.value,),
        )


def test_no_filters_compile_to_true():
    predicate = compile_filters([], page_alias="p")

    assert predicate.clause.as_string() == "TRUE"
    assert predicate.params == ()


def test_filters_combine_with_and_and_preserve_parameter_order():
    predicate = compile_filters(
        [ExampleFilter("site_id", 3), ExampleFilter("published_at", "2025-01-01")],
        page_alias="p",
    )

    assert (
        predicate.clause.as_string() == '"p"."site_id" = %s AND "p"."published_at" = %s'
    )
    assert predicate.params == (3, "2025-01-01")
