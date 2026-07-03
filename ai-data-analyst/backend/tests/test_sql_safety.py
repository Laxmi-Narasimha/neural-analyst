import pytest

from app.services.sql_safety import (
    UnsafeSQLError,
    enforce_row_limit,
    quote_identifier,
    validate_dataset_sql,
    validate_readonly_sql,
)


def test_validate_readonly_sql_allows_select():
    assert validate_readonly_sql("select 1") == "select 1"


def test_validate_readonly_sql_allows_cte():
    q = "with t as (select 1 as x) select x from t"
    assert validate_readonly_sql(q) == q


@pytest.mark.parametrize(
    "q",
    [
        "update users set x=1",
        "delete from users",
        "insert into t values (1)",
        "drop table users",
        "select 1; select 2",
        "/* hi */ insert into t values (1)",
    ],
)
def test_validate_readonly_sql_rejects_unsafe(q):
    with pytest.raises(UnsafeSQLError):
        validate_readonly_sql(q)


@pytest.mark.parametrize(
    "q",
    [
        "select * from read_csv_auto('C:/secrets.csv')",
        "select * from parquet_scan('C:/secrets.parquet')",
        "select * from 'C:/secrets.parquet'",
        "with t as (select * from read_parquet('C:/x.parquet')) select * from t",
    ],
)
def test_validate_dataset_sql_rejects_file_exfil_patterns(q):
    with pytest.raises(UnsafeSQLError):
        validate_dataset_sql(q)


def test_validate_dataset_sql_allows_select_from_dataset_table():
    assert validate_dataset_sql("select count(*) as n from dataset") == "select count(*) as n from dataset"


def test_enforce_row_limit_wraps():
    out = enforce_row_limit("select 1 as x", 123)
    assert "LIMIT 123" in out
    assert out.lower().startswith("select * from (")


def test_quote_identifier_simple():
    assert quote_identifier("users", dialect="postgresql") == '"users"'
    assert quote_identifier("public.users", dialect="postgresql") == '"public"."users"'


def test_quote_identifier_rejects_invalid():
    with pytest.raises(ValueError):
        quote_identifier("users; drop table x", dialect="postgresql")
