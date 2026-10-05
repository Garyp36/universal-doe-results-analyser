"""Checks for the calculation core. Run with:  pytest -q"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import doe_core as dc  # noqa: E402


def five_level(n):
    return [dc.Factor(f"F{i}", "", [1, 2, 3, 4, 5]) for i in range(n)]


@pytest.fixture(scope="module")
def starter():
    table = pd.DataFrame({
        "Factor": ["Air inlet velocity", "Water inlet temperature", "Fin pitch", "Tube pitch"],
        "Unit": ["m/s", "K", "mm", "mm"],
        "Levels": ["2, 4, 6, 8, 10", "333.15 343.15 353.15 363.15 373.15", "2,3,4,5,6", "25; 30; 35; 40; 45"]})
    factors, errors = dc.factors_from_table(table)
    assert errors == []
    design, meta = dc.build_design(factors, "L25")
    return factors, design, meta


# ---------------------------------------------------------------- arrays and designs
@pytest.mark.parametrize("info", dc.oa_catalog(), ids=lambda i: i.key)
def test_every_catalog_array_is_orthogonal(info):
    arr = dc.oa_array(info.key)
    assert arr.shape == (info.runs, len(info.levels))
    dc.validate_oa(arr, info.levels)


def test_broken_array_is_rejected():
    arr = dc.oa_array("L9")
    arr[0, 1] = (arr[0, 1] + 1) % 3
    with pytest.raises(ValueError):
        dc.validate_oa(arr, (3,) * 4)


def test_parse_levels():
    assert dc.parse_levels("2, 4, 6") == [2.0, 4.0, 6.0]
    assert dc.parse_levels("2 4 6") == [2.0, 4.0, 6.0]
    assert dc.parse_levels("low; high") == ["low", "high"]
    assert dc.parse_levels("Louvered fin, Plain fin") == ["Louvered fin", "Plain fin"]
    for bad in ["5", "", "1, 1", ", ".join(map(str, range(12)))]:
        with pytest.raises(ValueError):
            dc.parse_levels(bad)


def test_factor_table_reports_every_problem():
    table = pd.DataFrame({"Factor": ["A", "a", "Run", "", "B"], "Unit": [""] * 5,
                          "Levels": ["1,2", "3,4", "1 2", "1,2", "x"]})
    factors, errors = dc.factors_from_table(table)
    assert [f.name for f in factors] == ["A"]
    assert len(errors) == 4


def test_compatible_arrays_prefer_fewest_runs_and_hide_dominated_ones():
    found = dc.compatible_arrays([5, 5, 5, 5])
    assert [a["key"] for a in found] == ["L25", "L50"]
    mixed = dc.compatible_arrays([2, 2, 2, 3])
    assert mixed[0]["key"] == "L9" and mixed[0]["dummy"] == 3
    assert dc.compatible_arrays([5] * 9)[0]["key"] == "L50"
    assert dc.compatible_arrays([5] * 15) == []


def test_build_design_is_balanced_and_orthogonal(starter):
    factors, design, meta = starter
    assert len(design) == 25 and meta["array"] == "L25 (5⁶)"
    diag = dc.design_diagnostics(design, factors)
    assert diag["orthogonal"] and diag["unbalanced"] == [] and diag["error_df"] == 8
    for f in factors:
        assert sorted(design[f.name].value_counts().tolist()) == [5] * 5


def test_dummy_levels_keep_orthogonality_but_flag_imbalance():
    f = [dc.Factor("A", "", [10, 20]), dc.Factor("B", "", [1, 2]), dc.Factor("C", "", ["x", "y", "z"])]
    design, meta = dc.build_design(f, "L9")
    assert set(meta["dummy"]) == {"A", "B"}
    diag = dc.design_diagnostics(design, f)
    assert diag["orthogonal"] and set(diag["unbalanced"]) == {"A", "B"}


def test_full_factorial_covers_every_combination():
    f = [dc.Factor("A", "", [1, 2, 3]), dc.Factor("B", "", ["x", "y"])]
    design, meta = dc.full_factorial(f)
    assert len(design) == 6 and dc.full_factorial_runs(f) == 6
    assert len(design[["A", "B"]].drop_duplicates()) == 6


def test_confounded_factors_are_detected():
    design = pd.DataFrame({dc.RUN: range(1, 7), "A": [1, 1, 2, 2, 3, 3], "B": [1, 1, 2, 2, 3, 3]})
    f = [dc.Factor("A", "", [1, 2, 3]), dc.Factor("B", "", [1, 2, 3])]
    assert dc.design_diagnostics(design, f)["aliased"]


# ---------------------------------------------------------------- statistics
def test_f_distribution_against_known_values():
    assert dc.f_sf(4.103, 2, 10) == pytest.approx(0.05, abs=1e-3)
    assert dc.f_sf(0.0, 3, 9) == pytest.approx(1.0)
    stats = pytest.importorskip("scipy.stats")
    for F, d1, d2 in [(1.0, 4, 20), (12.5, 1, 3), (0.2, 6, 40), (30, 3, 5)]:
        assert dc.f_sf(F, d1, d2) == pytest.approx(stats.f.sf(F, d1, d2), rel=1e-9)


def test_anova_matches_textbook_sums_of_squares(starter):
    factors, design, _ = starter
    rng = np.random.default_rng(0)
    y = (3 * design["Fin pitch"] + 0.5 * design["Tube pitch"] + rng.normal(0, 0.3, 25)).to_numpy()
    table, meta = dc.anova(design, factors, y)
    assert table["Contribution (%)"].sum() + meta["residual_pct"] == pytest.approx(100.0)
    assert meta["additive"] and not meta["saturated"] and table.iloc[0]["Factor"] == "Fin pitch"
    grand = y.mean()
    for f in factors:
        ss = sum(5 * (y[design[f.name] == v].mean() - grand) ** 2 for v in f.levels)
        assert table.set_index("Factor").loc[f.name, "SS"] == pytest.approx(ss)
    top = table.iloc[0]
    assert top["p-value"] < 1e-6 and top["Verdict"] == "Strong"


def test_anova_with_unbalanced_design_recovers_a_noise_free_effect():
    f = [dc.Factor("A", "", [10, 20]), dc.Factor("B", "", [1, 2]), dc.Factor("C", "", ["x", "y", "z"])]
    design, _ = dc.build_design(f, "L9")
    y = (design["A"] / 10 * 2 + (design["C"] == "z") * 3).to_numpy(float)
    table, meta = dc.anova(design, f, y)
    assert meta["residual_pct"] < 1e-8
    assert set(table["Factor"].head(2)) == {"A", "C"}
    assert table.set_index("Factor").loc["B", "Verdict"] == "Negligible"


def test_saturated_design_pools_the_weakest_factor():
    f = five_level(6)
    design, _ = dc.build_design(f, "L25")
    y = (2 * design["F0"] + design["F1"] + np.random.default_rng(1).normal(0, 0.1, 25)).to_numpy()
    table, meta = dc.anova(design, f, y)
    assert meta["saturated"] and meta["pooled"] and meta["df_error_used"] >= 2
    assert table.set_index("Factor").loc[meta["pooled"][0], "Verdict"] == "Pooled"
    assert np.isfinite(table.iloc[0]["F"])


def test_straight_line_factor_gets_one_degree_of_freedom():
    rng = np.random.default_rng(2)
    x1, x2 = rng.uniform(0, 1, 30), rng.uniform(0, 1, 30)
    design = pd.DataFrame({dc.RUN: range(1, 31), "X1": x1, "X2": x2})
    f = [dc.Factor("X1", "", sorted(x1), kind="lin"), dc.Factor("X2", "", sorted(x2), kind="lin")]
    y = 5 * x1 + 1 * x2 + rng.normal(0, 0.05, 30)
    table, meta = dc.anova(design, f, y)
    assert (table["df"] == 1).all() and table.iloc[0]["Factor"] == "X1"
    assert meta["r2_pct"] > 95
    opt = dc.predict_optimum(design, f, y, ("higher", None), table)
    assert opt["chosen"]["X1"] == pytest.approx(x1.max())


def test_anova_rejects_flat_or_tiny_responses(starter):
    factors, design, _ = starter
    with pytest.raises(ValueError, match="no variation"):
        dc.anova(design, factors, np.ones(25))
    y = np.full(25, np.nan)
    y[:3] = [1, 2, 3]
    with pytest.raises(ValueError, match="at least"):
        dc.anova(design, factors, y)


def test_partial_results_still_analyse(starter):
    factors, design, _ = starter
    y = (design["Fin pitch"] * 2.0).to_numpy(float).copy()
    y[[3, 9, 15]] = np.nan
    table, meta = dc.anova(design, factors, y)
    assert meta["n"] == 22 and table.iloc[0]["Factor"] == "Fin pitch"


def test_optimum_picks_best_level_and_finds_tested_run(starter):
    factors, design, _ = starter
    y = (design["Fin pitch"] * 10 - design["Tube pitch"]).to_numpy(float)
    table, _ = dc.anova(design, factors, y)
    hi = dc.predict_optimum(design, factors, y, ("higher", None), table)
    lo = dc.predict_optimum(design, factors, y, ("lower", None), table)
    assert hi["chosen"]["Fin pitch"] == 6 and hi["chosen"]["Tube pitch"] == 25
    assert lo["chosen"]["Fin pitch"] == 2 and lo["chosen"]["Tube pitch"] == 45
    assert hi["predicted"] == pytest.approx(10 * 6 - 25)
    near = dc.predict_optimum(design, factors, y, ("target", 30.0), table)
    assert abs(near["best_value"] - 30) == min(abs(y - 30))


# ---------------------------------------------------------------- comparison
def test_compare_metrics_known_values():
    m = dc.compare_metrics([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0])
    assert m["bias"] == pytest.approx(1.0) and m["rmse"] == pytest.approx(1.0)
    assert m["r"] == pytest.approx(1.0) and m["rho"] == pytest.approx(1.0) and m["slope"] == pytest.approx(1.0)
    assert dc.compare_metrics([1, 2], [1, 2]) is None
    m = dc.compare_metrics([1, 2, 3, np.nan], [1, 2, 4, 9])
    assert m["n"] == 3
    assert dc.agreement_rating(4.9) == "Good" and dc.agreement_rating(10) == "Moderate" and dc.agreement_rating(20) == "Poor"


def test_default_goal_guesses():
    assert dc.default_goal("Air pressure drop") == "lower"
    assert dc.default_goal("Heat transfer rate") == "higher"
    assert dc.default_goal("Coolant outlet temperature") == "lower"


# ---------------------------------------------------------------- GRA
def test_gra_grades_and_ranking(starter):
    factors, design, _ = starter
    demo = dc.demo_results(design, factors, "CFD")
    frame, units, _ = dc.parse_results(demo, design, factors)
    keys = ["Heat transfer rate", "Air pressure drop"]
    goals = {"Heat transfer rate": ("higher", None), "Air pressure drop": ("lower", None)}
    g = dc.gra_for(design, factors, frame, keys, goals, {k: 1 for k in keys})
    grades = g["grades"]
    assert ((grades > 0) & (grades <= 1)).all()
    assert sum(g["weights"].values()) == pytest.approx(1.0)
    assert g["ranking"].iloc[0][dc.RUN] == g["best_run"] == int(design[dc.RUN].iloc[np.argmax(grades)])
    assert g["ranking"]["GRA grade"].is_monotonic_decreasing


def test_gra_needs_enough_complete_runs(starter):
    factors, design, _ = starter
    frame = pd.DataFrame({dc.RUN: design[dc.RUN], "A": np.arange(25.0), "B": np.nan})
    frame.loc[:2, "B"] = [1, 2, 3]
    with pytest.raises(ValueError, match="at least"):
        dc.gra_for(design, factors, frame, ["A", "B"], {"A": ("higher", None), "B": ("higher", None)}, {"A": 1, "B": 1})


def test_target_goal_prefers_values_near_target():
    norm = dc.normalize_for_gra([1, 5, 9], ("target", 5.0))
    assert norm.argmax() == 1
    with pytest.raises(ValueError):
        dc.normalize_for_gra([1, 2], ("target", None))


# ---------------------------------------------------------------- reading tables
def test_parse_design_from_own_table():
    raw = pd.DataFrame({"Run No": [1, 2, 3, 4, 5, 6], "Velocity (m/s)": [1, 1, 2, 2, 3, 3],
                        "Type": list("abcabc"), "Heat (W)": [5.1, 5.2, 5.3, 5.4, 5.5, 5.6]})
    design, factors, notes = dc.parse_design(raw, ["Velocity (m/s)", "Type"])
    assert [f.name for f in factors] == ["Velocity", "Type"] and factors[0].unit == "m/s"
    assert factors[1].levels == ["a", "b", "c"]
    prof = dc.profile_columns(raw)
    assert prof.set_index("Column").loc["Heat (W)", "Distinct"] == 6


def test_parse_design_detects_many_valued_numeric_factors_and_bad_tables():
    x = np.linspace(0, 1, 20)
    design, factors, _ = dc.parse_design(pd.DataFrame({"X": x, "Y": np.arange(20) % 3}))
    assert {f.name: f.kind for f in factors} == {"X": "lin", "Y": "cat"}
    with pytest.raises(ValueError, match="too many"):
        dc.parse_design(pd.DataFrame({"X": x}), kinds={"X": "cat"})
    with pytest.raises(ValueError, match="empty cells"):
        dc.parse_design(pd.DataFrame({"X": [1, 2, None, 1, 2, 3], "Y": list("abcabc")}))
    with pytest.raises(ValueError, match="at least"):
        dc.parse_design(pd.DataFrame({"X": [1, 2, 3]}))


def test_parse_results_accepts_run_sheet_round_trip(starter):
    factors, design, _ = starter
    sheet = dc.run_sheet(design, factors, ["Heat transfer rate (W)", "Air pressure drop (Pa)"])
    with pytest.raises(ValueError, match="still empty"):
        dc.parse_results(sheet, design, factors)
    sheet.iloc[:, -2:] = np.random.default_rng(1).normal(100, 5, (25, 2))
    res, units, notes = dc.parse_results(sheet.sample(frac=1, random_state=3), design, factors)
    assert res[dc.RUN].tolist() == list(range(1, 26))
    assert units == {"Heat transfer rate": "W", "Air pressure drop": "Pa"}
    assert "Suggested order" not in res.columns and any("factor columns" in n for n in notes)


def test_parse_results_handles_partial_replicates_and_mistakes(starter):
    factors, design, _ = starter
    demo = dc.demo_results(design, factors, "CFD")
    res, _, notes = dc.parse_results(demo.iloc[:20], design, factors)
    assert res.shape[0] == 25 and res.iloc[20:, 1:].isna().all().all()
    assert any("20 of 25" in n for n in notes)
    rep = pd.concat([demo, demo.iloc[:5]])
    assert any("averaged" in n for n in dc.parse_results(rep, design, factors)[2])
    with pytest.raises(ValueError, match="rows are read in order"):
        dc.parse_results(demo.drop(columns=[dc.RUN]).iloc[:10], design, factors)
    with pytest.raises(ValueError, match="not in the design"):
        dc.parse_results(demo.assign(Run=range(100, 125)), design, factors)
    bad = demo.astype(object)
    bad.iloc[2, 1] = "n/a"
    with pytest.raises(ValueError, match="non-numeric"):
        dc.parse_results(bad, design, factors)


def test_merge_datasets_matches_names_and_flags_unit_clashes(starter):
    factors, design, _ = starter
    a, ua, _ = dc.parse_results(dc.demo_results(design, factors, "CFD"), design, factors)
    b_raw = dc.demo_results(design, factors, "Experiment").rename(
        columns={"Heat transfer rate (W)": "heat  TRANSFER rate (kW)"})
    b, ub, _ = dc.parse_results(b_raw, design, factors)
    merged, units, notes = dc.merge_datasets({"CFD": (a, ua), "Exp": (b, ub)})
    assert set(merged["CFD"].columns) == set(merged["Exp"].columns)
    assert units["Heat transfer rate"] == "W" and any("kW" in n for n in notes)


# ---------------------------------------------------------------- the whole chain
def test_full_analysis_findings_and_export(starter):
    factors, design, meta = starter
    items = {n: dc.parse_results(dc.demo_results(design, factors, n, seed=s), design, factors)[:2]
             for n, s in (("CFD", 7), ("Experiment", 8))}
    datasets, units, _ = dc.merge_datasets(items)
    selected = list(units)
    goals = {k: (dc.default_goal(k), None) for k in selected}
    result = dc.analyze(design, factors, datasets, selected, goals, reference="CFD",
                        gra={"keys": selected, "weights": {k: 1 for k in selected}})
    assert set(result["comparison"]["Experiment"]) == set(selected)
    assert set(result["gra"]) == {"CFD", "Experiment"} and not result["notes"]
    findings = dc.build_findings(design, factors, result, units, goals, "CFD", meta["array"])
    assert {f["section"] for f in findings} >= {"design", "compare", "drivers", "optimum", "gra", "caution"}
    assert ":" not in dc.strip_markup(":blue[**12 W**]")
    blob = dc.export_workbook(design, factors, datasets, result, findings, units)
    sheets = set(pd.ExcelFile(io.BytesIO(blob)).sheet_names)
    assert {"Findings", "Design", "Data CFD", "Comparison", "Factor contributions", "Level means",
            "Best settings", "GRA ranking"} <= sheets


def test_single_output_single_dataset_needs_no_comparison_or_gra(starter):
    factors, design, _ = starter
    res, units, _ = dc.parse_results(dc.demo_results(design, factors, "CFD"), design, factors)
    key = "Heat transfer rate"
    result = dc.analyze(design, factors, {"CFD": res}, [key], {key: ("higher", None)})
    assert result["comparison"] == {} and result["gra"] == {} and key in result["anova"]["CFD"]
    findings = dc.build_findings(design, factors, result, units, {key: ("higher", None)})
    assert not any(f["section"] in ("compare", "gra") for f in findings)


def test_gra_is_skipped_with_a_note_when_a_dataset_lacks_outputs(starter):
    factors, design, _ = starter
    full, _, _ = dc.parse_results(dc.demo_results(design, factors, "CFD"), design, factors)
    thin = full[[dc.RUN, "Heat transfer rate"]]
    keys = ["Heat transfer rate", "Air pressure drop"]
    goals = {k: ("higher", None) for k in keys}
    result = dc.analyze(design, factors, {"CFD": full, "Test": thin}, keys, goals, reference="CFD",
                        gra={"keys": keys, "weights": {k: 1 for k in keys}})
    assert set(result["gra"]) == {"CFD"}
    assert any("GRA was skipped for Test" in n for n in result["notes"])


def test_demo_results_are_reproducible_and_differ_between_sources(starter):
    factors, design, _ = starter
    a = dc.demo_results(design, factors, "CFD", seed=7)
    assert a.equals(dc.demo_results(design, factors, "CFD", seed=7))
    b = dc.demo_results(design, factors, "Experiment", seed=8)
    assert not np.allclose(a.iloc[:, 1], b.iloc[:, 1])
