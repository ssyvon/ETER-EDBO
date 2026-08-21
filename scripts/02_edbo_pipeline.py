#!/usr/bin/env python3
"""Build EDBO cleaned data, checks, tables, and metrics.

This script intentionally does not build charts. Notebooks own visual analysis;
the pipeline only creates processed CSV artifacts.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import pandas as pd


RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, filename: str) -> Path:
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    return path


def safe_divide(numerator, denominator):
    return numerator / denominator.replace({0: pd.NA})


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_excel(RAW_DIR / "edbo_raw.xlsx")
    df.columns = df.columns.str.strip()

    student_columns = [
        "Денна (бюджет)",
        "Денна (контракт)",
        "Заочна (бюджет)",
        "Заочна (контракт)",
        "Вечірня (бюджет)",
        "Вечірня (контракт)",
        "Сума по рядку",
        "Сума по закладу",
    ]
    for column in student_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    for column in ["Код", "Код головного закладу"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    budget_columns = ["Денна (бюджет)", "Заочна (бюджет)", "Вечірня (бюджет)"]
    contract_columns = ["Денна (контракт)", "Заочна (контракт)", "Вечірня (контракт)"]
    full_time_columns = ["Денна (бюджет)", "Денна (контракт)"]
    part_time_columns = ["Заочна (бюджет)", "Заочна (контракт)"]
    evening_columns = ["Вечірня (бюджет)", "Вечірня (контракт)"]

    df["budget_students"] = df[budget_columns].sum(axis=1)
    df["contract_students"] = df[contract_columns].sum(axis=1)
    df["full_time_students"] = df[full_time_columns].sum(axis=1)
    df["part_time_students"] = df[part_time_columns].sum(axis=1)
    df["evening_students"] = df[evening_columns].sum(axis=1)
    df["calculated_row_sum"] = df[budget_columns + contract_columns].sum(axis=1)
    df["row_sum_diff"] = df["Сума по рядку"] - df["calculated_row_sum"]
    df["row_sum_is_consistent"] = df["row_sum_diff"].eq(0)
    return df


def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    overview_metrics = pd.DataFrame(
        [
            {"metric": "rows", "value": len(df)},
            {"metric": "columns", "value": df.shape[1]},
            {"metric": "unique_institutions", "value": df["Код"].nunique()},
            {"metric": "unique_regions", "value": df["Регіон"].nunique()},
            {"metric": "unique_specialties", "value": df["Код спеціальності"].nunique()},
            {"metric": "unique_degrees", "value": df["Освітній ступінь"].nunique()},
            {"metric": "duplicate_rows", "value": df.duplicated().sum()},
            {"metric": "total_students_row_sum", "value": df["Сума по рядку"].sum()},
            {"metric": "budget_students", "value": df["budget_students"].sum()},
            {"metric": "contract_students", "value": df["contract_students"].sum()},
        ]
    )
    missing_values = df.isna().sum().rename("missing_count").reset_index()
    missing_values.columns = ["column", "missing_count"]
    missing_values["missing_share"] = missing_values["missing_count"] / len(df)
    missing_values = missing_values.sort_values("missing_count", ascending=False)

    region_summary = (
        df.groupby("Регіон", dropna=False)
        .agg(
            students=("Сума по рядку", "sum"),
            institutions=("Код", "nunique"),
            specialties=("Код спеціальності", "nunique"),
            budget_students=("budget_students", "sum"),
            contract_students=("contract_students", "sum"),
        )
        .reset_index()
    )
    region_summary["students_per_institution"] = safe_divide(region_summary["students"], region_summary["institutions"])
    region_summary["budget_share"] = safe_divide(region_summary["budget_students"], region_summary["students"])
    region_summary = region_summary.sort_values("students", ascending=False)

    institution_summary = (
        df.groupby(["Код", "Назва закладу освіти", "Регіон", "Форма власності", "Підпорядкування"], dropna=False)
        .agg(
            students=("Сума по рядку", "sum"),
            reported_institution_students=("Сума по закладу", "max"),
            rows=("Сума по рядку", "size"),
            degrees=("Освітній ступінь", "nunique"),
            specialties=("Код спеціальності", "nunique"),
            budget_students=("budget_students", "sum"),
            contract_students=("contract_students", "sum"),
        )
        .reset_index()
    )
    institution_summary["budget_share"] = safe_divide(institution_summary["budget_students"], institution_summary["students"])
    institution_summary["reported_minus_grouped_students"] = institution_summary["reported_institution_students"] - institution_summary["students"]
    institution_summary = institution_summary.sort_values("students", ascending=False)

    degree_summary = (
        df.groupby("Освітній ступінь", dropna=False)
        .agg(
            students=("Сума по рядку", "sum"),
            rows=("Сума по рядку", "size"),
            institutions=("Код", "nunique"),
            specialties=("Код спеціальності", "nunique"),
            budget_students=("budget_students", "sum"),
            contract_students=("contract_students", "sum"),
        )
        .reset_index()
    )
    degree_summary["student_share"] = degree_summary["students"] / degree_summary["students"].sum()
    degree_summary["budget_share"] = safe_divide(degree_summary["budget_students"], degree_summary["students"])
    degree_summary = degree_summary.sort_values("students", ascending=False)

    specialty_summary = (
        df.groupby(["Код спеціальності", "Назва спеціальності"], dropna=False)
        .agg(
            students=("Сума по рядку", "sum"),
            institutions=("Код", "nunique"),
            regions=("Регіон", "nunique"),
            degrees=("Освітній ступінь", "nunique"),
            budget_students=("budget_students", "sum"),
            contract_students=("contract_students", "sum"),
        )
        .reset_index()
    )
    specialty_summary["budget_share"] = safe_divide(specialty_summary["budget_students"], specialty_summary["students"])
    specialty_summary = specialty_summary.sort_values("students", ascending=False)

    budget_contract_summary = pd.DataFrame(
        {
            "funding_type": ["Бюджет", "Контракт"],
            "students": [df["budget_students"].sum(), df["contract_students"].sum()],
        }
    )
    budget_contract_summary["share"] = budget_contract_summary["students"] / budget_contract_summary["students"].sum()

    study_form_summary = pd.DataFrame(
        {
            "study_form": ["Денна", "Заочна", "Вечірня"],
            "students": [df["full_time_students"].sum(), df["part_time_students"].sum(), df["evening_students"].sum()],
        }
    )
    study_form_summary["share"] = study_form_summary["students"] / study_form_summary["students"].sum()

    ownership_summary = (
        df.groupby("Форма власності", dropna=False)
        .agg(students=("Сума по рядку", "sum"), institutions=("Код", "nunique"))
        .reset_index()
        .sort_values("students", ascending=False)
    )
    governing_summary = (
        df.groupby("Підпорядкування", dropna=False)
        .agg(students=("Сума по рядку", "sum"), institutions=("Код", "nunique"))
        .reset_index()
        .sort_values("students", ascending=False)
    )
    consistency_checks = pd.DataFrame(
        [
            {"check": "row_sum_consistent", "rows": int(df["row_sum_is_consistent"].sum())},
            {"check": "row_sum_inconsistent", "rows": int((~df["row_sum_is_consistent"]).sum())},
            {"check": "zero_row_sum", "rows": int(df["Сума по рядку"].eq(0).sum())},
            {"check": "negative_row_sum", "rows": int(df["Сума по рядку"].lt(0).sum())},
            {"check": "missing_region", "rows": int(df["Регіон"].isna().sum())},
            {"check": "missing_institution_name", "rows": int(df["Назва закладу освіти"].isna().sum())},
        ]
    )

    return {
        "edbo_overview_metrics.csv": overview_metrics,
        "edbo_missing_values.csv": missing_values,
        "edbo_region_summary.csv": region_summary,
        "edbo_institution_summary.csv": institution_summary,
        "edbo_degree_summary.csv": degree_summary,
        "edbo_specialty_summary.csv": specialty_summary,
        "edbo_budget_contract_summary.csv": budget_contract_summary,
        "edbo_study_form_summary.csv": study_form_summary,
        "edbo_ownership_summary.csv": ownership_summary,
        "edbo_governing_body_summary.csv": governing_summary,
        "edbo_consistency_checks.csv": consistency_checks,
        "edbo_inconsistent_row_sums.csv": df.loc[~df["row_sum_is_consistent"]].copy(),
    }


def main() -> None:
    ensure_dirs()
    df = load_and_prepare()
    df.to_csv(PROCESSED_DIR / "edbo_prepared.csv", index=False)
    tables = build_tables(df)
    for filename, table in tables.items():
        save_table(table, filename)

    overview = tables["edbo_overview_metrics.csv"].set_index("metric")["value"]
    print("EDBO pipeline complete")
    print(f"Rows: {int(overview['rows']):,}")
    print(f"Institutions: {int(overview['unique_institutions']):,}")
    print(f"Total students: {int(overview['total_students_row_sum']):,}")


if __name__ == "__main__":
    main()
