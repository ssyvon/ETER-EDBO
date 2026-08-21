#!/usr/bin/env python3
"""Build cleaned ETER data and derived country-level metrics.

This script intentionally does not build charts. Notebooks own visual analysis;
the pipeline only creates cleaned data, checks, tables, and reusable metrics.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd


RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPECIAL_CODES = ["a", "m", "x", "xc", "xr", "nc", "c", "s"]

STUDENT_LEVEL_COLS = ["students_isced5", "students_isced6", "students_isced7", "students_isced8"]
FOREIGN_LEVEL_COLS = [
    "foreign_students_isced5",
    "foreign_students_isced6",
    "foreign_students_isced7",
    "foreign_students_isced8",
]
STAFF_COLS = ["academic_personnel_fte", "total_personnel_fte", "support_admin_personnel_fte"]
FINANCE_COLS = [
    "total_current_expenditure_eur",
    "total_current_expenditure_ppp",
    "total_current_revenues_eur",
    "total_current_revenues_ppp",
    "third_party_funding_eur",
    "third_party_funding_ppp",
    "rd_expenditure_eur",
    "rd_expenditure_ppp",
]


SELECTED_COLUMNS = [
    "ETER ID Year",
    "ETER ID",
    "Institution Name",
    "English Institution Name",
    "Reference year",
    "Country Code",
    "Region of establishment (NUTS 2)",
    "Region of establishment (NUTS 3)",
    "Name of the city",
    "Geographic coordinates - latitude",
    "Geographic coordinates - longitude",
    "Multi-site institution",
    "Legal status",
    "Institution Category - English",
    "Institution Category standardised",
    "Total students enrolled at ISCED 5",
    "Total students enrolled at ISCED 6",
    "Total students enrolled at ISCED 7",
    "Total students enrolled ISCED 7 long degree",
    "Total students enrolled ISCED 5-7",
    "Total students enrolled at ISCED 8",
    "Students enrolled at ISCED 5 - foreigner",
    "Students enrolled at ISCED 6 - foreigner",
    "Students enrolled at ISCED 7 - foreigner",
    "Students enrolled ISCED 7 long degree - foreigner",
    "Students enrolled at ISCED 5-7 - foreigner",
    "Students enrolled at ISCED 8 - foreigner",
    "Total academic personnel (FTE)",
    "Total personnel (FTE)",
    "Number of support and administrative personnel (FTE)",
    "Total Current expenditure.1",
    "Total Current expenditure.2",
    "Total Current revenues.1",
    "Total Current revenues.2",
    "Total third party funding.1",
    "Total third party funding.2",
    "R&D Expenditure.1",
    "R&D Expenditure.2",
    "Flag Total students ISCED 5",
    "Flag Total students ISCED 6",
    "Flag Total students ISCED 7",
    "Flag Total students ISCED 7 long degree",
    "Flag Total students ISCED 5-7",
    "Flag Total students ISCED 8",
    "Flag Students ISCED 5 - citizenship",
    "Flag Students ISCED 6 - citizenship",
    "Flag Students ISCED 7 - citizenship",
    "Flag Students ISCED 7 long degree - citizenship",
    "Flag students enrolled at ISCED 5-7 - citizenship",
    "Flag Students ISCED 8 - citizenship",
    "Flag Total academic personnel (FTE)",
    "Flag Total personnel (FTE)",
    "Flag Number of support and administrative personnel (FTE)",
    "Flag Total current expenditure",
    "Flag Total current revenues",
    "Flag Total third party funding",
    "Flag R&D Expenditure",
]

RENAME_MAP = {
    "ETER ID Year": "record_id",
    "ETER ID": "institution_id",
    "Institution Name": "institution_name",
    "English Institution Name": "institution_name_en",
    "Reference year": "year",
    "Country Code": "country_code",
    "Region of establishment (NUTS 2)": "nuts2",
    "Region of establishment (NUTS 3)": "nuts3",
    "Name of the city": "city",
    "Geographic coordinates - latitude": "latitude",
    "Geographic coordinates - longitude": "longitude",
    "Multi-site institution": "multi_site",
    "Legal status": "legal_status",
    "Institution Category - English": "institution_category_en",
    "Institution Category standardised": "institution_category_std",
    "Total students enrolled at ISCED 5": "students_isced5",
    "Total students enrolled at ISCED 6": "students_isced6",
    "Total students enrolled at ISCED 7": "students_isced7",
    "Total students enrolled ISCED 7 long degree": "students_isced7_long_degree",
    "Total students enrolled ISCED 5-7": "students_isced5_7_total",
    "Total students enrolled at ISCED 8": "students_isced8",
    "Students enrolled at ISCED 5 - foreigner": "foreign_students_isced5",
    "Students enrolled at ISCED 6 - foreigner": "foreign_students_isced6",
    "Students enrolled at ISCED 7 - foreigner": "foreign_students_isced7",
    "Students enrolled ISCED 7 long degree - foreigner": "foreign_students_isced7_long_degree",
    "Students enrolled at ISCED 5-7 - foreigner": "foreign_students_isced5_7_total",
    "Students enrolled at ISCED 8 - foreigner": "foreign_students_isced8",
    "Total academic personnel (FTE)": "academic_personnel_fte",
    "Total personnel (FTE)": "total_personnel_fte",
    "Number of support and administrative personnel (FTE)": "support_admin_personnel_fte",
    "Total Current expenditure.1": "total_current_expenditure_eur",
    "Total Current expenditure.2": "total_current_expenditure_ppp",
    "Total Current revenues.1": "total_current_revenues_eur",
    "Total Current revenues.2": "total_current_revenues_ppp",
    "Total third party funding.1": "third_party_funding_eur",
    "Total third party funding.2": "third_party_funding_ppp",
    "R&D Expenditure.1": "rd_expenditure_eur",
    "R&D Expenditure.2": "rd_expenditure_ppp",
    "Flag Total students ISCED 5": "flag_students_isced5",
    "Flag Total students ISCED 6": "flag_students_isced6",
    "Flag Total students ISCED 7": "flag_students_isced7",
    "Flag Total students ISCED 7 long degree": "flag_students_isced7_long_degree",
    "Flag Total students ISCED 5-7": "flag_students_isced5_7_total",
    "Flag Total students ISCED 8": "flag_students_isced8",
    "Flag Students ISCED 5 - citizenship": "flag_foreign_students_isced5",
    "Flag Students ISCED 6 - citizenship": "flag_foreign_students_isced6",
    "Flag Students ISCED 7 - citizenship": "flag_foreign_students_isced7",
    "Flag Students ISCED 7 long degree - citizenship": "flag_foreign_students_isced7_long_degree",
    "Flag students enrolled at ISCED 5-7 - citizenship": "flag_foreign_students_isced5_7_total",
    "Flag Students ISCED 8 - citizenship": "flag_foreign_students_isced8",
    "Flag Total academic personnel (FTE)": "flag_academic_personnel_fte",
    "Flag Total personnel (FTE)": "flag_total_personnel_fte",
    "Flag Number of support and administrative personnel (FTE)": "flag_support_admin_personnel_fte",
    "Flag Total current expenditure": "flag_total_current_expenditure",
    "Flag Total current revenues": "flag_total_current_revenues",
    "Flag Total third party funding": "flag_third_party_funding",
    "Flag R&D Expenditure": "flag_rd_expenditure",
}


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def safe_divide(numerator, denominator) -> pd.Series:
    numerator = pd.Series(numerator, index=getattr(denominator, "index", None), dtype="float64")
    denominator = pd.Series(denominator, index=numerator.index, dtype="float64")
    return numerator.where(denominator.gt(0)) / denominator.where(denominator.gt(0))


def row_sum_min_count(dataframe: pd.DataFrame, columns: list[str], min_count: int = 1) -> pd.Series:
    return dataframe[columns].sum(axis=1, min_count=min_count)


def sum_min_count(series: pd.Series):
    return series.sum(min_count=1)


def positive_or_nan(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values > 0)


def clean_eter_data() -> pd.DataFrame:
    raw_path = RAW_DIR / "ETER_fullDump_27042023.csv"
    df_raw = pd.read_csv(raw_path, sep=";", encoding="utf-8-sig", decimal=",", low_memory=False)

    missing_selected = [col for col in SELECTED_COLUMNS if col not in df_raw.columns]
    if missing_selected:
        raise ValueError(f"Missing selected ETER columns: {missing_selected}")

    df_base = df_raw[SELECTED_COLUMNS].copy().rename(columns=RENAME_MAP)
    df_base.to_csv(PROCESSED_DIR / "df_base.csv", index=False, encoding="utf-8-sig")

    special_codes = pd.read_excel(RAW_DIR / "CorrespondenceTable_NamesCodesToLabels.xlsx", sheet_name="SpecialCodes")
    special_codes.to_csv(PROCESSED_DIR / "special_codes.csv", index=False, encoding="utf-8-sig")

    flag_cols = [col for col in df_base.columns if col.startswith("flag_")]
    non_flag_cols = [col for col in df_base.columns if col not in flag_cols]
    df_base[non_flag_cols] = df_base[non_flag_cols].replace(
        r"^\s*(a|m|x|xc|xr|nc|c|s)\s*$",
        pd.NA,
        regex=True,
    )

    numeric_float_cols = [
        "students_isced5",
        "students_isced6",
        "students_isced7",
        "students_isced7_long_degree",
        "students_isced5_7_total",
        "students_isced8",
        "foreign_students_isced5",
        "foreign_students_isced6",
        "foreign_students_isced7",
        "foreign_students_isced7_long_degree",
        "foreign_students_isced5_7_total",
        "foreign_students_isced8",
        "academic_personnel_fte",
        "total_personnel_fte",
        "support_admin_personnel_fte",
        "total_current_expenditure_eur",
        "total_current_expenditure_ppp",
        "total_current_revenues_eur",
        "total_current_revenues_ppp",
        "third_party_funding_eur",
        "third_party_funding_ppp",
        "rd_expenditure_eur",
        "rd_expenditure_ppp",
        "latitude",
        "longitude",
    ]
    df_base[numeric_float_cols] = (
        df_base[numeric_float_cols]
        .astype("string")
        .apply(lambda col: col.str.strip().str.replace(",", ".", regex=False))
        .apply(pd.to_numeric, errors="coerce")
    )

    text_cols = df_base.select_dtypes(include=["object", "string"]).columns
    df_base[text_cols] = df_base[text_cols].astype("string").apply(lambda col: col.str.strip())
    df_base.to_csv(PROCESSED_DIR / "df_base_clean.csv", index=False, encoding="utf-8-sig")
    return df_base


def add_institution_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_metrics = df.copy()

    df_metrics["students_isced_components_observed"] = df_metrics[STUDENT_LEVEL_COLS].notna().sum(axis=1)
    df_metrics["students_total_calc"] = row_sum_min_count(df_metrics, STUDENT_LEVEL_COLS)
    df_metrics["students_total_reported_5_7_only"] = df_metrics["students_isced5_7_total"].where(
        df_metrics["students_isced5_7_total"].notna() & df_metrics["students_isced8"].isna()
    )
    df_metrics["students_total_reported_plus_isced8"] = (
        df_metrics["students_isced5_7_total"] + df_metrics["students_isced8"]
    ).where(df_metrics["students_isced5_7_total"].notna() & df_metrics["students_isced8"].notna())
    df_metrics["students_total_analysis_source"] = np.select(
        [
            df_metrics["students_total_reported_plus_isced8"].notna(),
            df_metrics["students_total_reported_5_7_only"].notna(),
            df_metrics["students_total_calc"].notna(),
        ],
        ["reported_5_7_plus_observed_8", "reported_5_7_only", "component_sum_5_8_observed"],
        default="missing",
    )
    df_metrics["students_total_analysis"] = (
        df_metrics["students_total_reported_plus_isced8"]
        .combine_first(df_metrics["students_total_reported_5_7_only"])
        .combine_first(df_metrics["students_total_calc"])
    )
    df_metrics["students_total_includes_isced8"] = (
        df_metrics["students_total_reported_plus_isced8"].notna()
        | (df_metrics["students_total_calc"].notna() & df_metrics["students_isced8"].notna())
    )
    df_metrics["students_total_component_sum_complete"] = (
        df_metrics["students_total_analysis_source"].eq("component_sum_5_8_observed")
        & df_metrics["students_isced_components_observed"].eq(len(STUDENT_LEVEL_COLS))
    )
    df_metrics["students_total_component_sum_incomplete"] = (
        df_metrics["students_total_analysis_source"].eq("component_sum_5_8_observed")
        & df_metrics["students_isced_components_observed"].between(1, len(STUDENT_LEVEL_COLS) - 1)
    )
    df_metrics["students_total_component_sum_incomplete_value"] = df_metrics["students_total_analysis"].where(
        df_metrics["students_total_component_sum_incomplete"]
    )

    df_metrics["foreign_students_isced_components_observed"] = df_metrics[FOREIGN_LEVEL_COLS].notna().sum(axis=1)
    df_metrics["foreign_students_total_calc"] = row_sum_min_count(df_metrics, FOREIGN_LEVEL_COLS)
    df_metrics["foreign_students_total_reported_5_7_only"] = df_metrics["foreign_students_isced5_7_total"].where(
        df_metrics["foreign_students_isced5_7_total"].notna() & df_metrics["foreign_students_isced8"].isna()
    )
    df_metrics["foreign_students_total_reported_plus_isced8"] = (
        df_metrics["foreign_students_isced5_7_total"] + df_metrics["foreign_students_isced8"]
    ).where(df_metrics["foreign_students_isced5_7_total"].notna() & df_metrics["foreign_students_isced8"].notna())
    df_metrics["foreign_students_total_analysis_source"] = np.select(
        [
            df_metrics["foreign_students_total_reported_plus_isced8"].notna(),
            df_metrics["foreign_students_total_reported_5_7_only"].notna(),
            df_metrics["foreign_students_total_calc"].notna(),
        ],
        ["reported_5_7_plus_observed_8", "reported_5_7_only", "component_sum_5_8_observed"],
        default="missing",
    )
    df_metrics["foreign_students_total_analysis"] = (
        df_metrics["foreign_students_total_reported_plus_isced8"]
        .combine_first(df_metrics["foreign_students_total_reported_5_7_only"])
        .combine_first(df_metrics["foreign_students_total_calc"])
    )
    df_metrics["foreign_total_includes_isced8"] = (
        df_metrics["foreign_students_total_reported_plus_isced8"].notna()
        | (df_metrics["foreign_students_total_calc"].notna() & df_metrics["foreign_students_isced8"].notna())
    )
    df_metrics["foreign_total_component_sum_complete"] = (
        df_metrics["foreign_students_total_analysis_source"].eq("component_sum_5_8_observed")
        & df_metrics["foreign_students_isced_components_observed"].eq(len(FOREIGN_LEVEL_COLS))
    )
    df_metrics["foreign_total_component_sum_incomplete"] = (
        df_metrics["foreign_students_total_analysis_source"].eq("component_sum_5_8_observed")
        & df_metrics["foreign_students_isced_components_observed"].between(1, len(FOREIGN_LEVEL_COLS) - 1)
    )
    df_metrics["foreign_total_component_sum_incomplete_value"] = df_metrics["foreign_students_total_analysis"].where(
        df_metrics["foreign_total_component_sum_incomplete"]
    )

    df_metrics["has_complete_isced_structure"] = df_metrics[STUDENT_LEVEL_COLS].notna().all(axis=1)
    complete_isced_denominator = df_metrics["students_total_calc"].where(df_metrics["has_complete_isced_structure"])
    for level in [5, 6, 7, 8]:
        df_metrics[f"isced{level}_share"] = safe_divide(df_metrics[f"students_isced{level}"], complete_isced_denominator)

    df_metrics["phd_share"] = df_metrics["isced8_share"]
    df_metrics["foreign_students_share"] = safe_divide(
        df_metrics["foreign_students_total_analysis"],
        df_metrics["students_total_analysis"],
    )
    df_metrics["student_staff_ratio"] = safe_divide(
        df_metrics["students_total_analysis"],
        df_metrics["academic_personnel_fte"],
    )
    df_metrics["revenue_per_student_eur"] = safe_divide(df_metrics["total_current_revenues_eur"], df_metrics["students_total_analysis"])
    df_metrics["revenue_per_student_ppp"] = safe_divide(df_metrics["total_current_revenues_ppp"], df_metrics["students_total_analysis"])
    df_metrics["expenditure_per_student_eur"] = safe_divide(df_metrics["total_current_expenditure_eur"], df_metrics["students_total_analysis"])
    df_metrics["expenditure_per_student_ppp"] = safe_divide(df_metrics["total_current_expenditure_ppp"], df_metrics["students_total_analysis"])
    df_metrics["rd_expenditure_per_student_eur"] = safe_divide(df_metrics["rd_expenditure_eur"], df_metrics["students_total_analysis"])
    df_metrics["rd_expenditure_per_student_ppp"] = safe_divide(df_metrics["rd_expenditure_ppp"], df_metrics["students_total_analysis"])

    quality_checks = pd.Series(
        {
            "negative_student_values": df_metrics[STUDENT_LEVEL_COLS + ["students_isced5_7_total"]].lt(0).any(axis=1).sum(),
            "negative_foreign_student_values": df_metrics[FOREIGN_LEVEL_COLS + ["foreign_students_isced5_7_total"]].lt(0).any(axis=1).sum(),
            "negative_staff_values": df_metrics[STAFF_COLS].lt(0).any(axis=1).sum(),
            "negative_finance_values": df_metrics[FINANCE_COLS].lt(0).any(axis=1).sum(),
            "students_total_analysis_zero_or_negative": df_metrics["students_total_analysis"].le(0).sum(),
            "students_total_without_observed_isced8": (~df_metrics["students_total_includes_isced8"] & df_metrics["students_total_analysis"].notna()).sum(),
            "students_total_component_sum_incomplete": df_metrics["students_total_component_sum_incomplete"].sum(),
            "foreign_total_without_observed_isced8": (~df_metrics["foreign_total_includes_isced8"] & df_metrics["foreign_students_total_analysis"].notna()).sum(),
            "foreign_total_component_sum_incomplete": df_metrics["foreign_total_component_sum_incomplete"].sum(),
            "foreign_share_above_1": df_metrics["foreign_students_share"].gt(1).sum(),
            "isced_share_above_1_any": df_metrics[["isced5_share", "isced6_share", "isced7_share", "isced8_share"]].gt(1).any(axis=1).sum(),
            "student_staff_ratio_above_100": df_metrics["student_staff_ratio"].gt(100).sum(),
            "student_staff_ratio_below_1": df_metrics["student_staff_ratio"].between(0, 1, inclusive="neither").sum(),
            "expenditure_per_student_ppp_above_200k": df_metrics["expenditure_per_student_ppp"].gt(200_000).sum(),
        }
    )
    quality_checks_df = quality_checks.reset_index()
    quality_checks_df.columns = ["check", "affected_rows"]

    anomaly_rules = {
        "negative_student_values": df_metrics[STUDENT_LEVEL_COLS + ["students_isced5_7_total"]].lt(0).any(axis=1),
        "negative_foreign_student_values": df_metrics[FOREIGN_LEVEL_COLS + ["foreign_students_isced5_7_total"]].lt(0).any(axis=1),
        "negative_staff_values": df_metrics[STAFF_COLS].lt(0).any(axis=1),
        "negative_finance_values": df_metrics[FINANCE_COLS].lt(0).any(axis=1),
        "students_total_analysis_zero_or_negative": df_metrics["students_total_analysis"].le(0),
        "students_total_without_observed_isced8": ~df_metrics["students_total_includes_isced8"] & df_metrics["students_total_analysis"].notna(),
        "students_total_component_sum_incomplete": df_metrics["students_total_component_sum_incomplete"],
        "foreign_total_without_observed_isced8": ~df_metrics["foreign_total_includes_isced8"] & df_metrics["foreign_students_total_analysis"].notna(),
        "foreign_total_component_sum_incomplete": df_metrics["foreign_total_component_sum_incomplete"],
        "foreign_share_above_1": df_metrics["foreign_students_share"].gt(1),
        "isced_share_above_1_any": df_metrics[["isced5_share", "isced6_share", "isced7_share", "isced8_share"]].gt(1).any(axis=1),
        "student_staff_ratio_above_100": df_metrics["student_staff_ratio"].gt(100),
        "student_staff_ratio_below_1": df_metrics["student_staff_ratio"].between(0, 1, inclusive="neither"),
        "expenditure_per_student_ppp_above_200k": df_metrics["expenditure_per_student_ppp"].gt(200_000),
    }
    id_cols = [col for col in ["record_id", "institution_id", "institution_name", "country_code", "city", "year"] if col in df_metrics.columns]
    value_cols = [
        "students_total_analysis",
        "students_total_analysis_source",
        "students_total_includes_isced8",
        "students_total_component_sum_incomplete",
        "foreign_students_total_analysis",
        "foreign_students_total_analysis_source",
        "foreign_total_includes_isced8",
        "foreign_total_component_sum_incomplete",
        "foreign_students_share",
        "student_staff_ratio",
        "expenditure_per_student_ppp",
    ]
    student_anomalies = []
    finance_anomalies = []
    for rule_name, mask in anomaly_rules.items():
        rows = df_metrics.loc[mask.fillna(False), id_cols + [col for col in value_cols if col in df_metrics.columns]].copy()
        rows.insert(0, "anomaly", rule_name)
        if rule_name.startswith(("negative_finance", "student_staff", "expenditure")):
            finance_anomalies.append(rows)
        else:
            student_anomalies.append(rows)

    student_anomalies_df = pd.concat(student_anomalies, ignore_index=True) if student_anomalies else pd.DataFrame()
    finance_anomalies_df = pd.concat(finance_anomalies, ignore_index=True) if finance_anomalies else pd.DataFrame()
    return df_metrics, quality_checks_df, student_anomalies_df, finance_anomalies_df


def add_pairwise_ratio_metrics(source: pd.DataFrame, target: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    result = target.copy()
    group_cols = ["country_code", "year"]
    group_sizes = (
        source.groupby(group_cols, dropna=False)["record_id"]
        .count()
        .rename("_records_total_for_pairwise")
        .reset_index()
    )
    result = result.merge(group_sizes, on=group_cols, how="left")

    for spec in specs:
        metric = spec["metric"]
        numerator = spec["numerator"]
        denominator = spec["denominator"]
        aggregate_numerator = spec.get("aggregate_numerator")
        aggregate_denominator = spec.get("aggregate_denominator")
        if aggregate_numerator and aggregate_denominator:
            result[f"{metric}_aggregate_unmatched"] = safe_divide(result[aggregate_numerator], result[aggregate_denominator])

        valid = source[numerator].notna() & positive_or_nan(source[denominator]).notna()
        pairwise = (
            source.loc[valid]
            .groupby(group_cols, dropna=False)
            .agg(
                **{
                    f"{metric}_pair_numerator": (numerator, sum_min_count),
                    f"{metric}_pair_denominator": (denominator, sum_min_count),
                    f"{metric}_pair_records": ("record_id", "count"),
                    f"{metric}_pair_institutions": ("institution_id", "nunique"),
                }
            )
            .reset_index()
        )
        result = result.merge(pairwise, on=group_cols, how="left")
        result[metric] = safe_divide(result[f"{metric}_pair_numerator"], result[f"{metric}_pair_denominator"])
        result[f"{metric}_pair_records"] = result[f"{metric}_pair_records"].fillna(0).astype("int64")
        result[f"{metric}_pair_institutions"] = result[f"{metric}_pair_institutions"].fillna(0).astype("int64")
        result[f"{metric}_pair_record_coverage"] = safe_divide(
            result[f"{metric}_pair_records"],
            result["_records_total_for_pairwise"],
        )
        if aggregate_denominator:
            result[f"{metric}_denominator_coverage"] = safe_divide(
                result[f"{metric}_pair_denominator"],
                result[aggregate_denominator],
            ).clip(upper=1)

    return result.drop(columns=["_records_total_for_pairwise"])


def build_country_metrics(df_metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    country_base = (
        df_metrics.groupby(["country_code", "year"], dropna=False)
        .agg(
            records_count=("record_id", "nunique"),
            institutions_count=("institution_id", "nunique"),
            total_students=("students_total_analysis", sum_min_count),
            students_total_calc=("students_total_calc", sum_min_count),
            students_total_with_observed_isced8=("students_total_includes_isced8", "sum"),
            students_total_component_sum_complete_records=("students_total_component_sum_complete", "sum"),
            students_total_component_sum_incomplete_records=("students_total_component_sum_incomplete", "sum"),
            students_total_component_sum_incomplete_students=("students_total_component_sum_incomplete_value", sum_min_count),
            students_isced5=("students_isced5", sum_min_count),
            students_isced6=("students_isced6", sum_min_count),
            students_isced7=("students_isced7", sum_min_count),
            students_isced8=("students_isced8", sum_min_count),
            foreign_students_total=("foreign_students_total_analysis", sum_min_count),
            foreign_students_total_with_observed_isced8=("foreign_total_includes_isced8", "sum"),
            foreign_total_component_sum_complete_records=("foreign_total_component_sum_complete", "sum"),
            foreign_total_component_sum_incomplete_records=("foreign_total_component_sum_incomplete", "sum"),
            foreign_total_component_sum_incomplete_students=("foreign_total_component_sum_incomplete_value", sum_min_count),
            foreign_students_isced5=("foreign_students_isced5", sum_min_count),
            foreign_students_isced6=("foreign_students_isced6", sum_min_count),
            foreign_students_isced7=("foreign_students_isced7", sum_min_count),
            foreign_students_isced8=("foreign_students_isced8", sum_min_count),
            academic_personnel_fte=("academic_personnel_fte", sum_min_count),
            total_personnel_fte=("total_personnel_fte", sum_min_count),
            support_admin_personnel_fte=("support_admin_personnel_fte", sum_min_count),
            total_revenues_eur=("total_current_revenues_eur", sum_min_count),
            total_revenues_ppp=("total_current_revenues_ppp", sum_min_count),
            total_expenditure_eur=("total_current_expenditure_eur", sum_min_count),
            total_expenditure_ppp=("total_current_expenditure_ppp", sum_min_count),
            third_party_funding_eur=("third_party_funding_eur", sum_min_count),
            third_party_funding_ppp=("third_party_funding_ppp", sum_min_count),
            rd_expenditure_eur=("rd_expenditure_eur", sum_min_count),
            rd_expenditure_ppp=("rd_expenditure_ppp", sum_min_count),
            mean_institution_size=("students_total_analysis", "mean"),
            median_institution_size=("students_total_analysis", "median"),
            min_institution_size=("students_total_analysis", "min"),
            max_institution_size=("students_total_analysis", "max"),
            cities_count=("city", "nunique"),
            records_with_complete_isced_structure=("has_complete_isced_structure", "sum"),
        )
        .reset_index()
    )
    student_coverage = (
        df_metrics.loc[df_metrics["students_total_analysis"].notna()]
        .groupby(["country_code", "year"], dropna=False)["institution_id"]
        .nunique()
        .rename("institutions_with_student_data")
        .reset_index()
    )
    country_profiles = country_base.merge(student_coverage, on=["country_code", "year"], how="left")
    country_profiles["institutions_with_student_data"] = country_profiles["institutions_with_student_data"].fillna(0).astype("int64")
    country_profiles["complete_isced_structure_share"] = safe_divide(
        country_profiles["records_with_complete_isced_structure"],
        country_profiles["records_count"],
    )
    country_profiles["isced_shares_reliable"] = country_profiles["complete_isced_structure_share"].ge(0.8)
    country_profiles["isced_structure_coverage_flag"] = pd.cut(
        country_profiles["complete_isced_structure_share"],
        bins=[-np.inf, 0, 0.5, 0.8, np.inf],
        labels=["no_complete_records", "low", "medium", "high"],
    )
    country_profiles["student_coverage_share"] = safe_divide(
        country_profiles["institutions_with_student_data"],
        country_profiles["institutions_count"],
    )
    country_profiles["students_total_component_sum_incomplete_share"] = safe_divide(
        country_profiles["students_total_component_sum_incomplete_students"],
        country_profiles["total_students"],
    ).fillna(0)
    country_profiles["foreign_total_component_sum_incomplete_share"] = safe_divide(
        country_profiles["foreign_total_component_sum_incomplete_students"],
        country_profiles["foreign_students_total"],
    ).fillna(0)
    country_profiles["students_total_reliable"] = country_profiles["students_total_component_sum_incomplete_share"].le(0.05)
    country_profiles["total_students_reliable"] = country_profiles["total_students"].where(country_profiles["students_total_reliable"])

    country_has_all_isced_totals = country_profiles[STUDENT_LEVEL_COLS].notna().all(axis=1)
    country_isced_denominator = country_profiles["students_total_calc"].where(country_has_all_isced_totals)
    for level in [5, 6, 7, 8]:
        observed_share = safe_divide(country_profiles[f"students_isced{level}"], country_isced_denominator)
        country_profiles[f"isced{level}_share_observed"] = observed_share
        country_profiles[f"isced{level}_share_reliable"] = observed_share.where(country_profiles["isced_shares_reliable"])
        country_profiles[f"isced{level}_share"] = country_profiles[f"isced{level}_share_reliable"]
        country_profiles[f"foreign_students_isced{level}_share"] = safe_divide(
            country_profiles[f"foreign_students_isced{level}"],
            country_profiles[f"students_isced{level}"],
        )

    country_profiles["phd_share"] = country_profiles["isced8_share"]
    country_profiles["phd_share_observed"] = country_profiles["isced8_share_observed"]
    country_profiles["phd_share_reliable"] = country_profiles["isced8_share_reliable"]
    country_profiles["avg_observed_institution_size"] = safe_divide(
        country_profiles["total_students"],
        country_profiles["institutions_with_student_data"],
    )
    country_profiles["students_per_registered_institution"] = safe_divide(
        country_profiles["total_students"],
        country_profiles["institutions_count"],
    )
    country_profiles["avg_institution_size"] = country_profiles["students_per_registered_institution"]

    pairwise_specs = [
        {"metric": "foreign_students_share", "numerator": "foreign_students_total_analysis", "denominator": "students_total_analysis", "aggregate_numerator": "foreign_students_total", "aggregate_denominator": "total_students"},
        {"metric": "student_staff_ratio", "numerator": "students_total_analysis", "denominator": "academic_personnel_fte", "aggregate_numerator": "total_students", "aggregate_denominator": "academic_personnel_fte"},
        {"metric": "revenue_per_student_eur", "numerator": "total_current_revenues_eur", "denominator": "students_total_analysis", "aggregate_numerator": "total_revenues_eur", "aggregate_denominator": "total_students"},
        {"metric": "revenue_per_student_ppp", "numerator": "total_current_revenues_ppp", "denominator": "students_total_analysis", "aggregate_numerator": "total_revenues_ppp", "aggregate_denominator": "total_students"},
        {"metric": "expenditure_per_student_eur", "numerator": "total_current_expenditure_eur", "denominator": "students_total_analysis", "aggregate_numerator": "total_expenditure_eur", "aggregate_denominator": "total_students"},
        {"metric": "expenditure_per_student_ppp", "numerator": "total_current_expenditure_ppp", "denominator": "students_total_analysis", "aggregate_numerator": "total_expenditure_ppp", "aggregate_denominator": "total_students"},
        {"metric": "third_party_funding_per_student_ppp", "numerator": "third_party_funding_ppp", "denominator": "students_total_analysis", "aggregate_numerator": "third_party_funding_ppp", "aggregate_denominator": "total_students"},
        {"metric": "rd_expenditure_per_student_ppp", "numerator": "rd_expenditure_ppp", "denominator": "students_total_analysis", "aggregate_numerator": "rd_expenditure_ppp", "aggregate_denominator": "total_students"},
        {"metric": "rd_expenditure_share_of_expenditure_ppp", "numerator": "rd_expenditure_ppp", "denominator": "total_current_expenditure_ppp", "aggregate_numerator": "rd_expenditure_ppp", "aggregate_denominator": "total_expenditure_ppp"},
    ]
    country_profiles = add_pairwise_ratio_metrics(df_metrics, country_profiles, pairwise_specs)
    country_profiles = country_profiles.sort_values(["country_code", "year"]).reset_index(drop=True)

    descriptive_metrics = [
        "institutions_count",
        "institutions_with_student_data",
        "student_coverage_share",
        "total_students",
        "avg_observed_institution_size",
        "students_per_registered_institution",
        "median_institution_size",
        "isced5_share",
        "isced6_share",
        "isced7_share",
        "isced8_share",
        "foreign_students_share",
        "student_staff_ratio",
        "revenue_per_student_ppp",
        "expenditure_per_student_ppp",
        "rd_expenditure_per_student_ppp",
    ]
    country_descriptive_stats = country_profiles[descriptive_metrics].describe(percentiles=[0.25, 0.5, 0.75]).T.rename(columns={"50%": "median"})
    metric_availability = (
        country_profiles[["country_code", "year"] + descriptive_metrics]
        .set_index(["country_code", "year"])
        .notna()
        .agg(["sum", "mean"])
        .T
        .rename(columns={"sum": "country_years_available", "mean": "share_available"})
        .sort_values("share_available", ascending=False)
    )

    latest_year = int(country_profiles["year"].max())
    latest_country_profiles = country_profiles[country_profiles["year"].eq(latest_year)].copy()
    ranking_metrics = ["total_students", "isced8_share", "foreign_students_share", "student_staff_ratio", "expenditure_per_student_ppp"]
    ranking_exclusions = []
    for metric in ranking_metrics:
        available = latest_country_profiles.dropna(subset=[metric])
        ranking_exclusions.append(
            {
                "latest_year": latest_year,
                "countries_total_in_dataset": country_profiles["country_code"].nunique(dropna=True),
                "countries_available_latest_year": latest_country_profiles["country_code"].nunique(dropna=True),
                "metric": metric,
                "excluded_countries": len(latest_country_profiles) - len(available),
                "available_countries": len(available),
            }
        )

    exclusion_rules = []
    for metric in descriptive_metrics:
        missing_mask = country_profiles[metric].isna()
        if missing_mask.any():
            excluded = country_profiles.loc[missing_mask, ["country_code", "year"]].copy()
            excluded["metric"] = metric
            excluded["exclusion_reason"] = "metric_missing_or_unreliable_after_coverage_filter"
            exclusion_rules.append(excluded)
    low_student_coverage = country_profiles.loc[country_profiles["student_coverage_share"].lt(0.8), ["country_code", "year", "student_coverage_share"]].copy()
    low_student_coverage["metric"] = "student_coverage_share"
    low_student_coverage["exclusion_reason"] = "student_coverage_below_80pct"
    low_isced_coverage = country_profiles.loc[~country_profiles["isced_shares_reliable"], ["country_code", "year", "complete_isced_structure_share"]].copy()
    low_isced_coverage["metric"] = "isced_structure"
    low_isced_coverage["exclusion_reason"] = "complete_isced_structure_below_80pct"
    exclusion_rules.extend([low_student_coverage, low_isced_coverage])

    outputs = {
        "country_descriptive_stats": country_descriptive_stats,
        "metric_availability": metric_availability,
        "ranking_exclusions": pd.DataFrame(ranking_exclusions),
        "country_metric_exclusions": pd.concat(exclusion_rules, ignore_index=True) if exclusion_rules else pd.DataFrame(),
    }
    return country_profiles, outputs


def build_overview_tables(df: pd.DataFrame, country_profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    loaded = {
        "df_base_clean": df,
        "country_profiles": country_profiles,
    }
    shape_report = pd.DataFrame(
        [
            {
                "table": name,
                "rows": table.shape[0],
                "columns": table.shape[1],
                "country_count": table["country_code"].nunique(dropna=True) if "country_code" in table.columns else np.nan,
                "year_min": table["year"].min() if "year" in table.columns else np.nan,
                "year_max": table["year"].max() if "year" in table.columns else np.nan,
            }
            for name, table in loaded.items()
        ]
    )

    overview_df = df.copy()
    overview_df["students_total_from_components"] = overview_df[STUDENT_LEVEL_COLS].sum(axis=1, min_count=1)
    overview_df["students_total_from_aggregate"] = overview_df[["students_isced5_7_total", "students_isced8"]].sum(axis=1, min_count=1)
    overview_df["students_total_overview"] = overview_df["students_total_from_aggregate"].combine_first(overview_df["students_total_from_components"])
    overview_df["foreign_students_total_from_components"] = overview_df[FOREIGN_LEVEL_COLS].sum(axis=1, min_count=1)
    overview_df["foreign_students_total_from_aggregate"] = overview_df[["foreign_students_isced5_7_total", "foreign_students_isced8"]].sum(axis=1, min_count=1)
    overview_df["foreign_students_total_overview"] = overview_df["foreign_students_total_from_aggregate"].combine_first(overview_df["foreign_students_total_from_components"])

    country_overview = (
        overview_df.groupby(["country_code", "year"], dropna=False)
        .agg(
            records=("record_id", "count"),
            institutions_count=("institution_id", "nunique"),
            total_students=("students_total_overview", sum_min_count),
            students_isced5_7_total=("students_isced5_7_total", sum_min_count),
            students_isced8=("students_isced8", sum_min_count),
            mean_observed_institution_size=("students_total_overview", "mean"),
            median_institution_size=("students_total_overview", "median"),
            min_institution_size=("students_total_overview", "min"),
            max_institution_size=("students_total_overview", "max"),
            foreign_students=("foreign_students_total_overview", sum_min_count),
            academic_personnel_fte=("academic_personnel_fte", sum_min_count),
            total_personnel_fte=("total_personnel_fte", sum_min_count),
            total_current_revenues_eur=("total_current_revenues_eur", sum_min_count),
            total_current_expenditure_eur=("total_current_expenditure_eur", sum_min_count),
            total_current_revenues_ppp=("total_current_revenues_ppp", sum_min_count),
            total_current_expenditure_ppp=("total_current_expenditure_ppp", sum_min_count),
        )
        .reset_index()
    )
    student_coverage = (
        overview_df.loc[overview_df["students_total_overview"].notna()]
        .groupby(["country_code", "year"], dropna=False)["institution_id"]
        .nunique()
        .rename("institutions_with_student_data")
        .reset_index()
    )
    country_overview = country_overview.merge(student_coverage, on=["country_code", "year"], how="left")
    country_overview["institutions_with_student_data"] = country_overview["institutions_with_student_data"].fillna(0).astype("int64")
    country_overview["student_coverage_share"] = safe_divide(country_overview["institutions_with_student_data"], country_overview["institutions_count"])
    country_overview["avg_observed_institution_size"] = safe_divide(country_overview["total_students"], country_overview["institutions_with_student_data"])
    country_overview["students_per_registered_institution"] = safe_divide(country_overview["total_students"], country_overview["institutions_count"])
    country_overview["avg_institution_size"] = country_overview["students_per_registered_institution"]
    country_overview["foreign_students_share"] = safe_divide(country_overview["foreign_students"], country_overview["total_students"])
    country_overview["student_staff_ratio"] = safe_divide(country_overview["total_students"], country_overview["academic_personnel_fte"])
    country_overview["revenue_per_student_eur"] = safe_divide(country_overview["total_current_revenues_eur"], country_overview["total_students"])
    country_overview["expenditure_per_student_eur"] = safe_divide(country_overview["total_current_expenditure_eur"], country_overview["total_students"])
    country_overview["revenue_per_student_ppp"] = safe_divide(country_overview["total_current_revenues_ppp"], country_overview["total_students"])
    country_overview["expenditure_per_student_ppp"] = safe_divide(country_overview["total_current_expenditure_ppp"], country_overview["total_students"])

    quality_checks = []
    for col in [col for col in overview_df.columns if col.startswith("students_") or col.startswith("foreign_students_")]:
        quality_checks.append({"check": f"negative values in {col}", "affected_rows": int((overview_df[col] < 0).sum())})
    for col in STAFF_COLS:
        quality_checks.append({"check": f"negative values in {col}", "affected_rows": int((overview_df[col] < 0).sum())})
    for col in [col for col in overview_df.columns if col.endswith("_eur") or col.endswith("_ppp")]:
        quality_checks.append({"check": f"negative values in {col}", "affected_rows": int((overview_df[col] < 0).sum())})
    quality_checks.extend(
        [
            {"check": "zero students_total_overview", "affected_rows": int((overview_df["students_total_overview"] == 0).sum())},
            {"check": "missing students_total_overview", "affected_rows": int(overview_df["students_total_overview"].isna().sum())},
            {"check": "foreign students above total students", "affected_rows": int((overview_df["foreign_students_total_overview"] > overview_df["students_total_overview"]).sum())},
        ]
    )
    overview_data_quality = pd.DataFrame(quality_checks).sort_values("affected_rows", ascending=False)
    return shape_report, country_overview, overview_data_quality


def save_country_metric_outputs(country_profiles: pd.DataFrame, outputs: dict[str, pd.DataFrame], quality_checks: pd.DataFrame, student_anomalies: pd.DataFrame, finance_anomalies: pd.DataFrame) -> None:
    country_profiles.to_csv(PROCESSED_DIR / "country_profiles.csv", index=False)

    quality_checks.to_csv(PROCESSED_DIR / "student_structure_quality_checks.csv", index=False)
    outputs["country_descriptive_stats"].to_csv(PROCESSED_DIR / "country_metrics_descriptive_stats.csv", index_label="metric")
    outputs["metric_availability"].to_csv(PROCESSED_DIR / "country_metric_availability.csv", index_label="metric")
    student_anomalies.to_csv(PROCESSED_DIR / "student_structure_anomalies.csv", index=False)
    finance_anomalies.to_csv(PROCESSED_DIR / "finance_staff_anomalies.csv", index=False)
    outputs["country_metric_exclusions"].to_csv(PROCESSED_DIR / "country_metric_exclusions.csv", index=False)
    outputs["ranking_exclusions"].to_csv(PROCESSED_DIR / "country_ranking_exclusions.csv", index=False)


def main() -> None:
    ensure_dirs()
    df_clean = clean_eter_data()
    df_metrics, quality_checks, student_anomalies, finance_anomalies = add_institution_metrics(df_clean)
    df_metrics.to_csv(PROCESSED_DIR / "eter_institution_metrics.csv", index=False)

    country_profiles, country_outputs = build_country_metrics(df_metrics)
    save_country_metric_outputs(country_profiles, country_outputs, quality_checks, student_anomalies, finance_anomalies)

    shape_report, country_overview, overview_quality = build_overview_tables(df_clean, country_profiles)
    shape_report.to_csv(PROCESSED_DIR / "eda_input_tables_overview.csv", index=False, encoding="utf-8-sig")
    country_overview.to_csv(PROCESSED_DIR / "country_overview.csv", index=False)
    overview_quality.to_csv(PROCESSED_DIR / "overview_data_quality.csv", index=False)

    print("ETER pipeline complete")
    print(f"Rows in df_base_clean: {len(df_clean):,}")
    print(f"Country-year rows: {len(country_profiles):,}")


if __name__ == "__main__":
    main()
