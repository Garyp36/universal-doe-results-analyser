"""Calculation core for the DOE Results Analyzer (no Streamlit code in here).

The app has no physics model. You define a design of experiments (factors and
levels), run it yourself in CFD or on the test rig, enter the results, and this
module does the statistics: comparison between sources, factor contribution
(ANOVA), level means, predicted best settings and grey relational analysis (GRA).
"""

from __future__ import annotations

import io
import itertools
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

RUN = "Run"
MAX_CAT_LEVELS = 8   # more distinct values than this are treated as a straight-line factor
MIN_RUNS = 5         # fewest valid results needed to analyse one output


# --------------------------------------------------------------------------
# Factors
# --------------------------------------------------------------------------
def fmt_level(v) -> str:
    """Short text for a level value: 2.0 -> '2', 333.15 -> '333.15'."""
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return f"{float(v):.10g}"
    return str(v)


@dataclass
class Factor:
    name: str
    unit: str = ""
    levels: list = field(default_factory=list)
    kind: str = "cat"   # "cat": compare level means; "lin": straight-line trend

    @property
    def numeric(self) -> bool:
        return bool(self.levels) and all(
            isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            for v in self.levels)

    @property
    def label(self) -> str:
        return f"{self.name} ({self.unit})" if self.unit else self.name

    def level_labels(self) -> list[str]:
        return [fmt_level(v) for v in self.levels]


_UNIT_RE = re.compile(r"^(.*?)\s*[\(\[]\s*([^()\[\]]*?)\s*[\)\]]\s*$")


def split_unit(header) -> tuple[str, str]:
    """'Air velocity (m/s)' -> ('Air velocity', 'm/s')."""
    h = str(header).strip()
    m = _UNIT_RE.match(h)
    if m and m.group(1).strip():
        return m.group(1).strip(), m.group(2).strip()
    return h, ""


def _norm(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def ulabel(name: str, units: dict | None) -> str:
    u = (units or {}).get(name, "")
    return f"{name} ({u})" if u else name


def parse_levels(text) -> list:
    """Turn '2, 4, 6' or '2 4 6' or 'low; high' into a list of numbers or text."""
    if text is None or (isinstance(text, float) and np.isnan(text)) or not str(text).strip():
        raise ValueError("enter at least two levels")
    s = str(text).strip()
    parts = re.split(r"[;,\n\t]+", s) if re.search(r"[;,\n\t]", s) else s.split()
    tokens = [t.strip() for t in parts if t.strip()]
    nums = []
    for t in tokens:
        try:
            nums.append(float(t))
        except ValueError:
            nums = None
            break
    levels = nums if nums is not None else tokens
    if len(levels) < 2:
        raise ValueError("enter at least two levels")
    keys = [round(v, 9) if isinstance(v, float) else v for v in levels]
    if len(set(keys)) != len(keys):
        raise ValueError("levels must all be different")
    if len(levels) > MAX_CAT_LEVELS:
        raise ValueError(f"up to {MAX_CAT_LEVELS} levels per factor")
    return levels


def factors_from_table(table: pd.DataFrame) -> tuple[list[Factor], list[str]]:
    """Read the editable factor table (columns Factor, Unit, Levels)."""
    factors, errors, seen = [], [], set()
    for i, row in table.reset_index(drop=True).iterrows():
        name = "" if pd.isna(row.get("Factor")) else str(row.get("Factor")).strip()
        lv = row.get("Levels")
        blank_levels = lv is None or (isinstance(lv, float) and np.isnan(lv)) or not str(lv).strip()
        if not name and blank_levels:
            continue
        where = f"Row {i + 1}"
        if not name:
            errors.append(f"{where}: give the factor a name.")
            continue
        if _norm(name) in seen or _norm(name) == _norm(RUN):
            errors.append(f"{where}: the name '{name}' is already used" + (" ('Run' is reserved)." if _norm(name) == _norm(RUN) else "."))
            continue
        try:
            levels = parse_levels(lv)
        except ValueError as exc:
            errors.append(f"{where} ({name}): {exc}.")
            continue
        seen.add(_norm(name))
        unit = "" if pd.isna(row.get("Unit")) else str(row.get("Unit")).strip()
        factors.append(Factor(name=name, unit=unit, levels=levels, kind="cat"))
    if not factors and not errors:
        errors.append("Add at least one factor with two or more levels.")
    return factors, errors


# --------------------------------------------------------------------------
# Orthogonal arrays
# --------------------------------------------------------------------------
_MUL4 = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]])   # GF(4) product


def build_oa(q: int, m: int) -> np.ndarray:
    """Linear orthogonal array with q**m runs and (q**m - 1)/(q - 1) columns (q = 2, 3, 4, 5, 7)."""
    if q not in (2, 3, 4, 5, 7):
        raise ValueError("q must be 2, 3, 4, 5 or 7")
    vecs = np.array(list(itertools.product(range(q), repeat=m)), dtype=int)
    pts = np.array([v for v in vecs if v.any() and v[np.flatnonzero(v)[0]] == 1])
    if q == 4:
        prod = _MUL4[vecs[:, None, :], pts[None, :, :]]
        return np.bitwise_xor.reduce(prod, axis=2)
    return (vecs @ pts.T) % q


def _plackett_burman(gen: str) -> np.ndarray:
    g = np.array([0 if c == "+" else 1 for c in gen])
    rows = [np.roll(g, s) for s in range(len(g))] + [np.ones(len(g), dtype=int)]
    return np.array(rows)


_L18_TEXT = """
1 1 1 1 1 1 1 1
1 1 2 2 2 2 2 2
1 1 3 3 3 3 3 3
1 2 1 1 2 2 3 3
1 2 2 2 3 3 1 1
1 2 3 3 1 1 2 2
1 3 1 2 1 3 2 3
1 3 2 3 2 1 3 1
1 3 3 1 3 2 1 2
2 1 1 3 3 2 2 1
2 1 2 1 1 3 3 2
2 1 3 2 2 1 1 3
2 2 1 2 3 1 3 2
2 2 2 3 1 2 1 3
2 2 3 1 2 3 2 1
2 3 1 3 2 3 1 2
2 3 2 1 3 1 2 3
2 3 3 2 1 2 3 1
"""

_L50_TEXT = """
1 1 1 1 1 1 1 1 1 1 1 1
1 1 2 2 2 2 2 2 2 2 2 2
1 1 3 3 3 3 3 3 3 3 3 3
1 1 4 4 4 4 4 4 4 4 4 4
1 1 5 5 5 5 5 5 5 5 5 5
1 2 1 2 3 4 5 1 2 3 4 5
1 2 2 3 4 5 1 2 3 4 5 1
1 2 3 4 5 1 2 3 4 5 1 2
1 2 4 5 1 2 3 4 5 1 2 3
1 2 5 1 2 3 4 5 1 2 3 4
1 3 1 3 5 2 4 4 1 3 5 2
1 3 2 4 1 3 5 5 2 4 1 3
1 3 3 5 2 4 1 1 3 5 2 4
1 3 4 1 3 5 2 2 4 1 3 5
1 3 5 2 4 1 3 3 5 2 4 1
1 4 1 4 2 5 3 5 3 1 4 2
1 4 2 5 3 1 4 1 4 2 5 3
1 4 3 1 4 2 5 2 5 3 1 4
1 4 4 2 5 3 1 3 1 4 2 5
1 4 5 3 1 4 2 4 2 5 3 1
1 5 1 5 4 3 2 4 3 2 1 5
1 5 2 1 5 4 3 5 4 3 2 1
1 5 3 2 1 5 4 1 5 4 3 2
1 5 4 3 2 1 5 2 1 5 4 3
1 5 5 4 3 2 1 3 2 1 5 4
2 1 1 1 4 5 4 3 2 5 2 3
2 1 2 2 5 1 5 4 3 1 3 4
2 1 3 3 1 2 1 5 4 2 4 5
2 1 4 4 2 3 2 1 5 3 5 1
2 1 5 5 3 4 3 2 1 4 1 2
2 2 1 2 1 3 3 2 4 5 5 4
2 2 2 3 2 4 4 3 5 1 1 5
2 2 3 4 3 5 5 4 1 2 2 1
2 2 4 5 4 1 1 5 2 3 3 2
2 2 5 1 5 2 2 1 3 4 4 3
2 3 1 3 3 1 2 5 5 4 2 4
2 3 2 4 4 2 3 1 1 5 3 5
2 3 3 5 5 3 4 2 2 1 4 1
2 3 4 1 1 4 5 3 3 2 5 2
2 3 5 2 2 5 1 4 4 3 1 3
2 4 1 4 5 4 1 2 5 2 3 3
2 4 2 5 1 5 2 3 1 3 4 4
2 4 3 1 2 1 3 4 2 4 5 5
2 4 4 2 3 2 4 5 3 5 1 1
2 4 5 3 4 3 5 1 4 1 2 2
2 5 1 5 2 2 5 3 4 4 3 1
2 5 2 1 3 3 1 4 5 5 4 2
2 5 3 2 4 4 2 5 1 1 5 3
2 5 4 3 5 5 3 1 2 2 1 4
2 5 5 4 1 1 4 2 3 3 2 5
"""


def _from_text(text: str) -> np.ndarray:
    return np.array([[int(v) - 1 for v in line.split()] for line in text.strip().splitlines()])


def validate_oa(arr, levels=None) -> None:
    """Raise ValueError unless every column is balanced and every pair of columns is orthogonal."""
    arr = np.asarray(arr)
    n, k = arr.shape
    levels = levels or tuple(int(arr[:, j].max()) + 1 for j in range(k))
    for j in range(k):
        counts = np.bincount(arr[:, j], minlength=levels[j])
        if len(counts) != levels[j] or len(set(counts)) != 1:
            raise ValueError(f"Column {j + 1} is not balanced across its {levels[j]} levels.")
    for a in range(k):
        for b in range(a + 1, k):
            table = np.zeros((levels[a], levels[b]), dtype=int)
            np.add.at(table, (arr[:, a], arr[:, b]), 1)
            if len(set(table.ravel())) != 1:
                raise ValueError(f"Columns {a + 1} and {b + 1} are not orthogonal.")


def _sup(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


@dataclass(frozen=True)
class OAInfo:
    key: str
    name: str
    runs: int
    levels: tuple   # level count of every column


def _spec(key, runs, levels, builder):
    counts: dict[int, int] = {}
    for q in levels:
        counts[q] = counts.get(q, 0) + 1
    name = f"{key.split('-')[0]} (" + " × ".join(f"{q}{_sup(c)}" for q, c in sorted(counts.items())) + ")"
    return key, OAInfo(key, name, runs, tuple(levels)), builder


_SPECS = {key: (info, builder) for key, info, builder in [
    _spec("L4", 4, (2,) * 3, lambda: build_oa(2, 2)),
    _spec("L8", 8, (2,) * 7, lambda: build_oa(2, 3)),
    _spec("L9", 9, (3,) * 4, lambda: build_oa(3, 2)),
    _spec("L12", 12, (2,) * 11, lambda: _plackett_burman("++-+++---+-")),
    _spec("L16", 16, (2,) * 15, lambda: build_oa(2, 4)),
    _spec("L16-4", 16, (4,) * 5, lambda: build_oa(4, 2)),
    _spec("L18", 18, (2,) + (3,) * 7, lambda: _from_text(_L18_TEXT)),
    _spec("L25", 25, (5,) * 6, lambda: build_oa(5, 2)),
    _spec("L27", 27, (3,) * 13, lambda: build_oa(3, 3)),
    _spec("L32", 32, (2,) * 31, lambda: build_oa(2, 5)),
    _spec("L49", 49, (7,) * 8, lambda: build_oa(7, 2)),
    _spec("L50", 50, (2,) + (5,) * 11, lambda: _from_text(_L50_TEXT)),
]}


def oa_catalog() -> list[OAInfo]:
    return sorted((info for info, _ in _SPECS.values()), key=lambda i: (i.runs, i.key))


@lru_cache(maxsize=None)
def _oa_cached(key: str) -> np.ndarray:
    info, builder = _SPECS[key]
    arr = np.asarray(builder(), dtype=int)
    assert arr.shape == (info.runs, len(info.levels)), key
    return arr


def oa_array(key: str) -> np.ndarray:
    return _oa_cached(key).copy()


def assign_columns(level_counts: list[int], col_levels: tuple) -> list[int] | None:
    """Give every factor the smallest free column with at least as many levels."""
    order = sorted(range(len(level_counts)), key=lambda i: -level_counts[i])
    free = list(range(len(col_levels)))
    out: list[int | None] = [None] * len(level_counts)
    for i in order:
        options = [c for c in free if col_levels[c] >= level_counts[i]]
        if not options:
            return None
        c = min(options, key=lambda c: (col_levels[c], c))
        out[i] = c
        free.remove(c)
    return out  # type: ignore[return-value]


def compatible_arrays(level_counts: list[int]) -> list[dict]:
    """Arrays that can carry the factors, smallest first."""
    found = []
    need = 1 + sum(n - 1 for n in level_counts)
    for info in oa_catalog():
        cols = assign_columns(level_counts, info.levels)
        if cols is None or info.runs < need:
            continue
        dummy = sum(1 for n, c in zip(level_counts, cols) if info.levels[c] > n)
        found.append({"key": info.key, "name": info.name, "runs": info.runs, "columns": cols,
                      "dummy": dummy, "error_df": info.runs - need,
                      "spare": len(info.levels) - len(level_counts)})
    found.sort(key=lambda d: (d["runs"], d["dummy"]))
    # hide an array when a smaller one carries the factors with fewer dummy levels
    return [b for b in found if not any(a["runs"] <= b["runs"] and a["dummy"] < b["dummy"] for a in found)]


def build_design(factors: list[Factor], oa_key: str) -> tuple[pd.DataFrame, dict]:
    """Fill the chosen array with your factor levels. Returns (design table, info)."""
    info, _ = _SPECS[oa_key]
    arr = oa_array(oa_key)
    counts = [len(f.levels) for f in factors]
    cols = assign_columns(counts, info.levels)
    if cols is None:
        raise ValueError(f"{info.name} cannot carry these factors.")
    data: dict = {RUN: np.arange(1, info.runs + 1)}
    dummy = []
    for f, c in zip(factors, cols):
        n = len(f.levels)
        idx = arr[:, c] % n          # extra array levels repeat the first levels (dummy levels)
        data[f.name] = [f.levels[k] for k in idx]
        if info.levels[c] > n:
            dummy.append(f.name)
    meta = {"array": info.name, "key": info.key, "runs": info.runs, "dummy": dummy,
            "spare": len(info.levels) - len(factors), "columns": dict(zip([f.name for f in factors], cols))}
    return pd.DataFrame(data), meta


def full_factorial(factors: list[Factor]) -> tuple[pd.DataFrame, dict]:
    """Every combination of every level (last factor changes fastest)."""
    combos = list(itertools.product(*[range(len(f.levels)) for f in factors]))
    data: dict = {RUN: np.arange(1, len(combos) + 1)}
    for j, f in enumerate(factors):
        data[f.name] = [f.levels[c[j]] for c in combos]
    meta = {"array": "Full factorial", "key": "FULL", "runs": len(combos), "dummy": [], "spare": 0, "columns": {}}
    return pd.DataFrame(data), meta


def full_factorial_runs(factors: list[Factor]) -> int:
    return int(np.prod([len(f.levels) for f in factors], dtype=float))


def randomized_order(n: int, seed: int = 1) -> np.ndarray:
    """A suggested execution order (1..n) for physical tests; CFD runs can ignore it."""
    return np.random.default_rng(seed).permutation(n) + 1


# --------------------------------------------------------------------------
# Reading tables
# --------------------------------------------------------------------------
_RUN_NAMES = {"run", "runno", "runnumber", "runid", "doerun", "exp", "expno", "experiment",
              "experimentno", "experimentnumber", "trial", "trialno", "srno", "sno", "slno", "serialno"}
_IGNORED = {"suggestedorder", "runorder", "executionorder", "order", "testorder", "notes", "remarks", "comment", "comments"}


def clean_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    keep = [i for i, c in enumerate(df.columns)
            if not (c == "" or c.lower().startswith("unnamed")) or df.iloc[:, i].notna().any()]
    df = df.iloc[:, keep]
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df.dropna(how="all").reset_index(drop=True)


def _to_numeric(s: pd.Series) -> pd.Series:
    """Numbers if the whole column reads as numbers (decimal commas allowed), else unchanged."""
    direct = pd.to_numeric(s, errors="coerce")
    if direct.notna().sum() == s.notna().sum():
        return direct
    as_text = s.astype("object").map(lambda v: v.replace(",", ".") if isinstance(v, str) and re.fullmatch(r"\s*-?\d+,\d+\s*", v) else v)
    comma = pd.to_numeric(as_text, errors="coerce")
    return comma if comma.notna().sum() == s.notna().sum() else s


def find_run_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if _norm(c) in _RUN_NAMES:
            return c
    return None


def profile_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per column: how many distinct values it holds and whether they are numbers."""
    df = clean_frame(raw)
    run_col = find_run_column(df)
    rows = []
    for c in df.columns:
        if c == run_col or _norm(c) in _IGNORED:
            continue
        s = _to_numeric(df[c])
        rows.append({"Column": c, "Distinct": int(s.dropna().nunique()),
                     "Numeric": bool(pd.api.types.is_numeric_dtype(s)), "Empty cells": int(s.isna().sum())})
    return pd.DataFrame(rows, columns=["Column", "Distinct", "Numeric", "Empty cells"])


def parse_design(raw: pd.DataFrame, factor_cols: list[str] | None = None,
                 kinds: dict | None = None) -> tuple[pd.DataFrame, list[Factor], list[str]]:
    """Read a design table you made yourself: one row per run, one column per factor."""
    df = clean_frame(raw)
    notes: list[str] = []
    if df.empty:
        raise ValueError("The design table is empty.")
    run_col = find_run_column(df)
    candidates = [c for c in df.columns if c != run_col and _norm(c) not in _IGNORED]
    cols = [c for c in (factor_cols if factor_cols is not None else candidates) if c in df.columns]
    if not cols:
        raise ValueError("No factor columns found. Give one column per factor, one row per run.")
    if len(df) < MIN_RUNS:
        raise ValueError(f"The design has {len(df)} runs; at least {MIN_RUNS} are needed.")
    if run_col is not None:
        runs = pd.to_numeric(df[run_col], errors="coerce")
        if runs.isna().any() or (runs % 1 != 0).any():
            raise ValueError(f"Column '{run_col}' must hold whole run numbers.")
        runs = runs.astype(int)
        if runs.duplicated().any():
            raise ValueError(f"Column '{run_col}' repeats run numbers: {sorted(set(runs[runs.duplicated()]))[:5]}.")
    else:
        runs = pd.Series(np.arange(1, len(df) + 1))
        notes.append("No run column found, so runs are numbered 1 to " + str(len(df)) + " in row order.")
    kinds = kinds or {}
    factors, data = [], {RUN: runs.to_numpy()}
    for c in cols:
        s = _to_numeric(df[c])
        if s.isna().any():
            rows = ", ".join(str(i + 1) for i in np.flatnonzero(s.isna())[:5])
            raise ValueError(f"Column '{c}' has empty cells (rows {rows}). Every run needs a value for every factor.")
        name, unit = split_unit(c)
        numeric = pd.api.types.is_numeric_dtype(s)
        vals = np.round(s.to_numpy(float), 9) if numeric else s.astype(str).str.strip().to_numpy()
        uniq = sorted(set(vals.tolist())) if numeric else list(dict.fromkeys(vals.tolist()))
        if len(uniq) < 2:
            notes.append(f"Column '{c}' has a single value, so it was ignored.")
            continue
        kind = kinds.get(c) or kinds.get(name) or ("lin" if numeric and len(uniq) > MAX_CAT_LEVELS else "cat")
        if kind == "lin" and not numeric:
            raise ValueError(f"'{c}' holds text, so it cannot be treated as a straight-line factor.")
        if kind == "cat" and len(uniq) > MAX_CAT_LEVELS:
            raise ValueError(f"'{c}' has {len(uniq)} different values, too many to compare as levels. "
                             "Treat it as a straight-line factor instead.")
        if _norm(name) in {_norm(f.name) for f in factors}:
            raise ValueError(f"Two factor columns are both called '{name}'.")
        factors.append(Factor(name=name, unit=unit, levels=list(uniq), kind=kind))
        data[name] = vals
    if not factors:
        raise ValueError("Every column has a single value, so there is nothing to analyse.")
    design = pd.DataFrame(data).sort_values(RUN).reset_index(drop=True)
    return design, factors, notes


def parse_results(raw: pd.DataFrame, design: pd.DataFrame, factors: list[Factor]
                  ) -> tuple[pd.DataFrame, dict, list[str]]:
    """Read result columns and line them up with the design runs.

    Returns (table with Run + one column per output, units, notes). Missing runs stay empty,
    repeated run numbers are averaged.
    """
    df = clean_frame(raw)
    notes: list[str] = []
    if df.empty:
        raise ValueError("The results table is empty.")
    run_col = find_run_column(df)
    factor_keys = {_norm(f.name): f for f in factors}
    out_cols, factor_cols = [], []
    for c in df.columns:
        if c == run_col:
            continue
        name, unit = split_unit(c)
        if _norm(name) in factor_keys or _norm(c) in factor_keys:
            factor_cols.append(c)
        elif _norm(name) in _IGNORED:
            continue
        else:
            out_cols.append((c, name, unit))
    if not out_cols:
        raise ValueError("No result columns found. Add one column per output (for example 'Heat transfer rate (W)').")

    values, units, empty = {}, {}, []
    for c, name, unit in out_cols:
        s = _to_numeric(df[c])
        if not pd.api.types.is_numeric_dtype(s):
            bad = df[c][pd.to_numeric(df[c], errors="coerce").isna() & df[c].notna()]
            raise ValueError(f"Column '{c}' has non-numeric text, for example '{bad.iloc[0]}' in row {bad.index[0] + 1}.")
        if s.notna().sum() == 0:
            empty.append(c)
            continue
        if name in values:
            raise ValueError(f"Two result columns are both called '{name}'.")
        values[name], units[name] = s.astype(float), unit
    if not values:
        raise ValueError("The result columns are still empty. Fill in the results for each run, then load the table again.")
    if empty:
        notes.append("Empty columns were skipped: " + ", ".join(empty) + ".")

    n_design = len(design)
    res = pd.DataFrame(values)
    if run_col is not None:
        runs = pd.to_numeric(df[run_col], errors="coerce")
        if runs.isna().any() or (runs % 1 != 0).any():
            raise ValueError(f"Column '{run_col}' must hold whole run numbers.")
        res.insert(0, RUN, runs.astype(int).to_numpy())
        unknown = sorted(set(res[RUN]) - set(design[RUN]))
        if unknown:
            raise ValueError(f"Run numbers {unknown[:6]} are not in the design (runs {int(design[RUN].min())} to {int(design[RUN].max())}).")
        dup = res[RUN][res[RUN].duplicated()].unique()
        if len(dup):
            notes.append(f"{len(dup)} run(s) had repeated rows (replicates) and were averaged.")
            res = res.groupby(RUN, as_index=False).mean()
    else:
        if len(res) != n_design:
            raise ValueError(f"There is no run column, so rows are read in order, but the table has {len(res)} rows "
                             f"and the design has {n_design} runs. Add a Run column or give all {n_design} rows.")
        res.insert(0, RUN, design[RUN].to_numpy())
        notes.append("No run column found; rows were matched to the design in order.")

    if factor_cols:
        check = df.copy()
        if run_col is not None and check[run_col].duplicated().any():
            check = check.drop_duplicates(run_col)
        mismatch = 0
        if run_col is not None:
            look = design.set_index(RUN)
            for c in factor_cols:
                f = factor_keys.get(_norm(split_unit(c)[0])) or factor_keys.get(_norm(c))
                if f is None:
                    continue
                theirs = _to_numeric(check[c])
                mine = look.loc[pd.to_numeric(check[run_col]).astype(int), f.name].to_numpy()
                a = theirs.to_numpy()
                if f.numeric and pd.api.types.is_numeric_dtype(theirs):
                    mismatch += int((~np.isclose(a.astype(float), mine.astype(float), equal_nan=True)).sum())
                else:
                    mismatch += int((a.astype(str) != mine.astype(str)).sum())
        notes.append("The factor columns in your table were ignored; the design table is used."
                     + (f" {mismatch} factor value(s) differ from the design, so check that the run numbers match."
                        if mismatch else ""))

    res = design[[RUN]].merge(res, on=RUN, how="left")
    filled = int(res[list(values)].notna().any(axis=1).sum())
    if filled < n_design:
        notes.append(f"Results cover {filled} of {n_design} runs; the other runs stay empty.")
    return res, units, notes


def merge_datasets(items: dict[str, tuple[pd.DataFrame, dict]]) -> tuple[dict[str, pd.DataFrame], dict, list[str]]:
    """Give the same output the same name in every dataset (matched ignoring case, spaces, punctuation)."""
    canon: dict[str, str] = {}
    units: dict[str, str] = {}
    notes: list[str] = []
    out: dict[str, pd.DataFrame] = {}
    for ds, (df, u) in items.items():
        rename = {}
        for col in df.columns:
            if col == RUN:
                continue
            key = _norm(col)
            if key not in canon:
                canon[key] = col
                units[col] = u.get(col, "")
            elif u.get(col, "") and units[canon[key]] and u.get(col, "") != units[canon[key]]:
                notes.append(f"{canon[key]} is in {units[canon[key]]} in one dataset and in {u.get(col)} in {ds}. Convert one so they match.")
            elif u.get(col, "") and not units[canon[key]]:
                units[canon[key]] = u.get(col, "")
            rename[col] = canon[key]
        out[ds] = df.rename(columns=rename)
    return out, units, notes


def run_sheet(design: pd.DataFrame, factors: list[Factor], outputs: list[str], seed: int = 1) -> pd.DataFrame:
    """The sheet to fill in: run, factor settings, suggested test order, empty output columns."""
    sheet = pd.DataFrame({RUN: design[RUN].to_numpy()})
    for f in factors:
        sheet[f.label] = design[f.name].to_numpy()
    sheet["Suggested order"] = randomized_order(len(design), seed)
    for o in outputs:
        sheet[o] = np.nan
    return sheet


# --------------------------------------------------------------------------
# Design checks
# --------------------------------------------------------------------------
def level_index(f: Factor, series) -> np.ndarray:
    """Position of every value in the factor's level list."""
    if f.numeric:
        x = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(float)
        lv = np.asarray(f.levels, float)
        idx = np.abs(x[:, None] - lv[None, :]).argmin(axis=1)
        if (~np.isclose(x, lv[idx], rtol=1e-9, atol=1e-12)).any():
            raise ValueError(f"{f.name} holds values that are not among its levels.")
        return idx
    lookup = {str(v): i for i, v in enumerate(f.levels)}
    idx = pd.Series(series).astype(str).map(lookup)
    if idx.isna().any():
        raise ValueError(f"{f.name} holds values that are not among its levels.")
    return idx.to_numpy(int)


def _block(f: Factor, series) -> np.ndarray:
    """Columns this factor adds to the regression (treatment dummies, or one standardized slope)."""
    if f.kind == "lin":
        x = pd.to_numeric(pd.Series(series)).to_numpy(float)
        sd = x.std()
        return ((x - x.mean()) / sd)[:, None] if sd > 0 else np.zeros((len(x), 1))
    idx = level_index(f, series)
    return np.column_stack([(idx == k).astype(float) for k in range(1, len(f.levels))])


def design_diagnostics(design: pd.DataFrame, factors: list[Factor]) -> dict:
    """Balance, orthogonality, confounding and the degrees of freedom left for error."""
    n = len(design)
    blocks = [_block(f, design[f.name]) for f in factors]
    X = np.hstack([np.ones((n, 1))] + blocks)
    rank = int(np.linalg.matrix_rank(X))
    diag = {"runs": n, "params": rank, "error_df": n - rank, "unbalanced": [], "aliased": [],
            "max_assoc": 0.0, "pair": None, "orthogonal": True}
    for j, f in enumerate(factors):
        if f.kind == "cat":
            counts = np.bincount(level_index(f, design[f.name]), minlength=len(f.levels))
            if len(set(counts)) > 1:
                diag["unbalanced"].append(f.name)
        rest = np.hstack([np.ones((n, 1))] + blocks[:j] + blocks[j + 1:])
        if rank - int(np.linalg.matrix_rank(rest)) < blocks[j].shape[1]:
            diag["aliased"].append(f.name)

    def codes(f):
        return level_index(f, design[f.name]) if f.kind == "cat" else pd.to_numeric(design[f.name]).to_numpy(float)

    for a, b in itertools.combinations(range(len(factors)), 2):
        fa, fb = factors[a], factors[b]
        if fa.kind == "cat" and fb.kind == "cat":
            tab = pd.crosstab(codes(fa), codes(fb)).to_numpy(float)
            exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / n
            chi2 = float(((tab - exp) ** 2 / exp).sum())
            dfm = min(tab.shape) - 1
            assoc = math.sqrt(chi2 / (n * dfm)) if dfm > 0 else 0.0
        else:
            ca, cb = codes(fa), codes(fb)
            assoc = 0.0 if ca.std() == 0 or cb.std() == 0 else abs(float(np.corrcoef(ca, cb)[0, 1]))
        if assoc > diag["max_assoc"]:
            diag["max_assoc"], diag["pair"] = assoc, (fa.name, fb.name)
    diag["orthogonal"] = diag["max_assoc"] < 1e-9
    return diag


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(ln_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(ln_front) * _betacf(b, a, 1.0 - x) / b


def f_sf(f_value: float, df1: float, df2: float) -> float:
    """P(F > f_value) for the F distribution."""
    if not np.isfinite(f_value) or df1 <= 0 or df2 <= 0 or f_value < 0:
        return float("nan")
    return float(_betainc(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * f_value)))


def verdict_from_p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "-"
    return "Strong" if p < 0.01 else "Significant" if p < 0.05 else "Weak" if p < 0.10 else "Not significant"


def anova(design: pd.DataFrame, factors: list[Factor], y, min_error_df: int = 2):
    """Main-effects ANOVA that works for balanced, unbalanced, saturated and partly filled designs.

    Each factor's sum of squares is the extra error left when that factor is dropped from the
    full model, so for an orthogonal design it equals the textbook value. When the design leaves
    fewer than `min_error_df` error degrees of freedom, the weakest factors are pooled into error
    (Taguchi practice) so that F values can still be formed.
    """
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    n = int(ok.sum())
    if n < MIN_RUNS:
        raise ValueError(f"only {n} valid result(s); at least {MIN_RUNS} are needed")
    yy = y[ok]
    if np.ptp(yy) <= 1e-12 * max(1.0, float(np.abs(yy).max())):
        raise ValueError("the response has no variation, so contributions are undefined")
    sst = float(((yy - yy.mean()) ** 2).sum())
    d = design.loc[ok].reset_index(drop=True)
    blocks = [_block(f, d[f.name]) for f in factors]
    ones = np.ones((n, 1))

    def fit(parts):
        X = np.hstack([ones] + parts)
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        r = yy - X @ beta
        return float(r @ r), int(np.linalg.matrix_rank(X))

    rss_full, rank_full = fit(blocks)
    rows, notes = [], []
    for j, f in enumerate(factors):
        rss_j, rank_j = fit(blocks[:j] + blocks[j + 1:])
        ss = rss_j - rss_full
        ss = 0.0 if ss < 1e-12 * sst else ss
        dfj = rank_full - rank_j
        if f.kind == "cat":
            idx = level_index(f, d[f.name])
            means = [yy[idx == k].mean() if (idx == k).any() else np.nan for k in range(len(f.levels))]
            spread = float(np.nanmax(means) - np.nanmin(means))
            kind_txt = f"{len(f.levels)} levels"
        else:
            x = pd.to_numeric(d[f.name]).to_numpy(float)
            slope = float(np.cov(x, yy, ddof=0)[0, 1] / x.var()) if x.var() > 0 else 0.0
            spread = abs(slope) * float(x.max() - x.min())
            kind_txt = "straight line"
        if dfj == 0:
            notes.append(f"{f.name} is fully mixed up (confounded) with other factors in this design, so it gets no share of its own.")
        rows.append({"Factor": f.name, "Type": kind_txt, "df": dfj, "SS": ss,
                     "MS": ss / dfj if dfj else 0.0, "Effect range": spread, "Pooled": False})

    df_err, ss_err = n - rank_full, max(rss_full, 0.0)
    pooled = []
    if df_err < min_error_df:
        for r in sorted([r for r in rows if r["df"] > 0], key=lambda r: r["MS"])[:-1]:
            if df_err >= min_error_df:
                break
            ss_err += r["SS"]
            df_err += r["df"]
            r["Pooled"] = True
            pooled.append(r["Factor"])
    ms_err = ss_err / df_err if df_err > 0 else float("nan")
    informative = np.isfinite(ms_err) and ss_err > 1e-10 * sst
    for r in rows:
        r["Contribution (%)"] = 100 * r["SS"] / sst
        if informative and r["df"] > 0 and not r["Pooled"]:
            r["F"] = r["MS"] / ms_err
            r["p-value"] = f_sf(r["F"], r["df"], df_err)
        else:
            r["F"], r["p-value"] = np.nan, np.nan
        r["Verdict"] = ("Pooled" if r["Pooled"] else "Negligible" if r["Contribution (%)"] < 1
                        else verdict_from_p(r["p-value"]))
    table = pd.DataFrame(rows).sort_values("Contribution (%)", ascending=False, kind="stable").reset_index(drop=True)
    table.insert(0, "Rank", np.arange(1, len(table) + 1))
    table = table[["Rank", "Factor", "Type", "df", "SS", "MS", "F", "p-value", "Contribution (%)",
                   "Effect range", "Verdict"]]
    total_ss = float(sum(r["SS"] for r in rows))
    resid_pct = 100 * max(rss_full, 0.0) / sst
    meta = {"n": n, "ss_total": sst, "ss_res": max(rss_full, 0.0), "df_res": n - rank_full,
            "df_error_used": df_err, "residual_pct": resid_pct, "r2_pct": 100 - resid_pct,
            "additive": abs(total_ss + max(rss_full, 0.0) - sst) <= 1e-6 * sst,
            "pooled": pooled, "saturated": n - rank_full == 0, "notes": notes,
            "has_f": bool(informative)}
    return table, meta


def level_means(design: pd.DataFrame, factors: list[Factor], y) -> pd.DataFrame:
    """Mean response at every level of every level-type factor (long table)."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    rows = []
    for f in factors:
        if f.kind != "cat":
            continue
        idx = level_index(f, design[f.name])
        for k, lab in enumerate(f.level_labels()):
            sel = (idx == k) & ok
            rows.append({"Factor": f.name, "Level": lab, "Level #": k + 1, "Value": f.levels[k],
                         "N": int(sel.sum()), "Mean": float(y[sel].mean()) if sel.any() else np.nan})
    return pd.DataFrame(rows, columns=["Factor", "Level", "Level #", "Value", "N", "Mean"])


def response_table(levels: pd.DataFrame, factors: list[Factor]) -> pd.DataFrame:
    """Taguchi-style response table: level means side by side, delta and rank."""
    rows = []
    for f in factors:
        sub = levels[levels["Factor"] == f.name]
        if sub.empty:
            continue
        r = {"Factor": f.name}
        for _, s in sub.iterrows():
            r[f"Level {int(s['Level #'])}"] = s["Mean"]
        r["Delta"] = float(sub["Mean"].max() - sub["Mean"].min())
        rows.append(r)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["Rank"] = t["Delta"].rank(ascending=False, method="min").astype(int)
    lvl_cols = sorted([c for c in t.columns if c.startswith("Level ")], key=lambda c: int(c.split()[1]))
    return t[["Factor"] + lvl_cols + ["Delta", "Rank"]].sort_values("Rank").reset_index(drop=True)


def goal_text(goal: tuple) -> str:
    mode, target = goal
    if mode == "target":
        return f"closest to {target:g}"
    return "higher is better" if mode == "higher" else "lower is better"


def default_goal(name: str) -> str:
    """Guess lower-is-better for losses and costs, higher-is-better otherwise."""
    low = re.search(r"drop|loss|resist|mass|weight|cost|pump|friction|drag|error|deviation|pressure|"
                    r"temperature out|outlet temp|tco|leak|noise|stress|vibration", name.lower())
    return "lower" if low else "higher"


def predict_optimum(design: pd.DataFrame, factors: list[Factor], y, goal: tuple,
                    table: pd.DataFrame, min_contrib_pct: float = 5.0) -> dict:
    """Best level of every factor and the additive prediction for that combination."""
    mode, target = goal
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    grand = float(y[ok].mean())
    contrib = dict(zip(table["Factor"], table["Contribution (%)"]))
    top = table["Factor"].iloc[0]

    def pick(values, current=None):
        values = np.asarray(values, float)
        if mode == "higher":
            return int(np.nanargmax(values))
        if mode == "lower":
            return int(np.nanargmin(values))
        return int(np.nanargmin(np.abs(values - target)))

    rows, chosen = [], {}
    for f in factors:
        counted = contrib.get(f.name, 0.0) >= min_contrib_pct or f.name == top
        if f.kind == "cat":
            idx = level_index(f, design[f.name])
            means = np.array([y[(idx == k) & ok].mean() if ((idx == k) & ok).any() else np.nan
                              for k in range(len(f.levels))])
            k = pick(grand + (means - grand))
            chosen[f.name] = f.levels[k]
            rows.append({"Factor": f.name, "Best level": fmt_level(f.levels[k]) + (f" {f.unit}" if f.unit else ""),
                         "Mean at that level": float(means[k]), "Effect vs overall mean": float(means[k] - grand),
                         "Counted in prediction": "yes" if counted else "no"})
        else:
            x = pd.to_numeric(design[f.name]).to_numpy(float)
            slope = float(np.cov(x[ok], y[ok], ddof=0)[0, 1] / x[ok].var()) if x[ok].var() > 0 else 0.0
            ends = np.array([x.min(), x.max()])
            k = pick(grand + slope * (ends - x[ok].mean()))
            chosen[f.name] = float(ends[k])
            rows.append({"Factor": f.name,
                         "Best level": fmt_level(ends[k]) + (f" {f.unit}" if f.unit else "") + (" (highest tested)" if k else " (lowest tested)"),
                         "Mean at that level": float(grand + slope * (ends[k] - x[ok].mean())),
                         "Effect vs overall mean": float(slope * (ends[k] - x[ok].mean())),
                         "Counted in prediction": "yes" if counted else "no"})
    predicted = grand + sum(r["Effect vs overall mean"] for r in rows if r["Counted in prediction"] == "yes")

    # does the chosen combination exist in the design? which tested run is best?
    match = np.ones(len(design), bool)
    for f in factors:
        v = design[f.name]
        match &= (np.isclose(pd.to_numeric(v, errors="coerce").to_numpy(float), float(chosen[f.name]))
                  if f.numeric else (v.astype(str) == str(chosen[f.name])).to_numpy())
    exists = int(design.loc[match & ok, RUN].iloc[0]) if (match & ok).any() else None
    yv = np.where(ok, y, np.nan)
    if mode == "higher":
        bi = int(np.nanargmax(yv))
    elif mode == "lower":
        bi = int(np.nanargmin(yv))
    else:
        bi = int(np.nanargmin(np.abs(yv - target)))
    return {"setting": pd.DataFrame(rows), "predicted": float(predicted), "grand": grand, "goal": goal,
            "tested_run": exists, "best_run": int(design[RUN].iloc[bi]), "best_value": float(yv[bi]),
            "chosen": chosen}


# --------------------------------------------------------------------------
# Comparison between sources
# --------------------------------------------------------------------------
def compare_metrics(reference, other) -> dict | None:
    """Paired metrics on runs where both values exist. Errors are other minus reference."""
    r = np.asarray(reference, float)
    o = np.asarray(other, float)
    ok = np.isfinite(r) & np.isfinite(o)
    if ok.sum() < 3:
        return None
    r, o = r[ok], o[ok]
    err = o - r
    rmse = float(np.sqrt(np.mean(err ** 2)))
    span = float(r.max() - r.min())
    pearson = float(np.corrcoef(r, o)[0, 1]) if r.std() > 0 and o.std() > 0 else None
    spearman = None
    if r.std() > 0 and o.std() > 0:
        v = pd.Series(r).rank().corr(pd.Series(o).rank())
        spearman = None if pd.isna(v) else float(v)
    slope = intercept = None
    if r.std() > 0:
        slope = float(np.cov(r, o, ddof=0)[0, 1] / r.var())
        intercept = float(o.mean() - slope * r.mean())
    nz = r != 0
    return {
        "n": int(ok.sum()), "bias": float(err.mean()), "mae": float(np.abs(err).mean()), "rmse": rmse,
        "max_abs_error": float(np.abs(err).max()),
        "nrmse_pct": None if span == 0 else 100 * rmse / span,
        "mape_pct": float(100 * np.mean(np.abs(err[nz] / r[nz]))) if nz.any() else None,
        "r": pearson, "rho": spearman, "slope": slope, "intercept": intercept,
        "bias_pct_of_ref_mean": float(100 * err.mean() / r.mean()) if r.mean() != 0 else None,
    }


def agreement_rating(nrmse_pct: float | None) -> str:
    if nrmse_pct is None:
        return "n/a"
    return "Good" if nrmse_pct < 5 else "Moderate" if nrmse_pct < 15 else "Poor"


# --------------------------------------------------------------------------
# Grey relational analysis
# --------------------------------------------------------------------------
def normalize_for_gra(values, goal: tuple) -> np.ndarray:
    mode, target = goal
    v = np.asarray(values, float)
    if mode == "target":
        if target is None:
            raise ValueError("a target value is needed for 'closest to target'")
        dev = np.abs(v - target)
        return np.ones_like(v) if dev.max() == 0 else 1 - dev / dev.max()
    lo, hi = v.min(), v.max()
    if hi == lo:
        return np.ones_like(v)
    return (hi - v) / (hi - lo) if mode == "lower" else (v - lo) / (hi - lo)


def calculate_gra(frame: pd.DataFrame, keys: list[str], goals: dict, weights: dict, zeta: float = 0.5):
    norm = {k: normalize_for_gra(frame[k], goals[k]) for k in keys}
    dev = {k: np.abs(1 - norm[k]) for k in keys}
    all_dev = np.concatenate([dev[k] for k in keys])
    dmin, dmax = all_dev.min(), all_dev.max()
    coef = ({k: np.ones(len(frame)) for k in keys} if dmax == 0
            else {k: (dmin + zeta * dmax) / (dev[k] + zeta * dmax) for k in keys})
    grades = sum(weights[k] * coef[k] for k in keys)
    return np.asarray(grades, float), coef


def gra_for(design: pd.DataFrame, factors: list[Factor], frame: pd.DataFrame, keys: list[str],
            goals: dict, weights: dict, zeta: float = 0.5, min_contrib_pct: float = 5.0) -> dict:
    """GRA on the runs where every selected output has a value, plus the factor analysis of the grade."""
    sub = frame[keys]
    ok = sub.notna().all(axis=1).to_numpy()
    if ok.sum() < MIN_RUNS:
        raise ValueError(f"only {int(ok.sum())} run(s) have all of {', '.join(keys)}; at least {MIN_RUNS} are needed")
    total = sum(float(weights[k]) for k in keys)
    w = {k: float(weights[k]) / total for k in keys}
    g_ok, coef_ok = calculate_gra(sub[ok], keys, goals, w, zeta)
    grades = np.full(len(frame), np.nan)
    grades[ok] = g_ok
    coef = {k: np.full(len(frame), np.nan) for k in keys}
    for k in keys:
        coef[k][ok] = coef_ok[k]
    table, meta = anova(design, factors, grades)
    levels = level_means(design, factors, grades)
    opt = predict_optimum(design, factors, grades, ("higher", None), table, min_contrib_pct)
    ranking = pd.DataFrame({RUN: design[RUN].to_numpy(), "GRA grade": grades})
    for k in keys:
        ranking[k] = frame[k].to_numpy()
    ranking = ranking.dropna(subset=["GRA grade"]).sort_values("GRA grade", ascending=False).reset_index(drop=True)
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
    return {"keys": keys, "goals": {k: goals[k] for k in keys}, "weights": w, "grades": grades, "coef": coef,
            "table": table, "meta": meta, "levels": levels, "optimum": opt, "ranking": ranking,
            "best_run": int(ranking.iloc[0][RUN]), "best_grade": float(ranking.iloc[0]["GRA grade"])}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def analyze(design: pd.DataFrame, factors: list[Factor], datasets: dict[str, pd.DataFrame],
            selected: list[str], goals: dict, reference: str | None = None,
            detailed: list[str] | None = None, gra: dict | None = None, zeta: float = 0.5) -> dict:
    """Run every analysis the user switched on and return the results in one dict.

    `goals` maps every selected output to ('higher'|'lower'|'target', target or None).
    `gra` is {'keys': [...], 'weights': {...}} or None. `detailed` lists the datasets that get
    the factor, best-setting and GRA steps (default: all).
    """
    res: dict = {"coverage": {}, "comparison": {}, "anova": {}, "levels": {}, "optimum": {},
                 "gra": {}, "notes": [], "diagnostics": design_diagnostics(design, factors)}
    detailed = list(datasets) if detailed is None else detailed
    for name, df in datasets.items():
        res["coverage"][name] = {k: int(df[k].notna().sum()) for k in selected if k in df.columns}

    if reference in datasets:
        ref = datasets[reference]
        for name, df in datasets.items():
            if name == reference:
                continue
            per = {}
            for k in selected:
                if k in df.columns and k in ref.columns:
                    m = compare_metrics(ref[k], df[k])
                    if m is None:
                        res["notes"].append(f"{k}: fewer than 3 runs have values in both {reference} and {name}, so they were not compared.")
                    else:
                        per[k] = m
            res["comparison"][name] = per

    for name in detailed:
        df = datasets[name]
        res["anova"][name], res["levels"][name], res["optimum"][name] = {}, {}, {}
        for k in selected:
            if k not in df.columns:
                continue
            try:
                table, meta = anova(design, factors, df[k].to_numpy())
            except ValueError as exc:
                res["notes"].append(f"Factor analysis skipped for {k} in {name}: {exc}.")
                continue
            res["anova"][name][k] = (table, meta)
            res["levels"][name][k] = level_means(design, factors, df[k].to_numpy())
            res["optimum"][name][k] = predict_optimum(design, factors, df[k].to_numpy(), goals[k], table)
        if gra:
            have = [k for k in gra["keys"] if k in df.columns and df[k].notna().sum() >= MIN_RUNS]
            if len(have) < 2:
                res["notes"].append(f"GRA was skipped for {name}: it holds {len(have)} of the {len(gra['keys'])} chosen outputs "
                                    "with enough results, and GRA needs at least two.")
                continue
            try:
                res["gra"][name] = gra_for(design, factors, df, have, goals, gra["weights"], zeta)
            except ValueError as exc:
                res["notes"].append(f"GRA skipped for {name}: {exc}.")
    return res


# --------------------------------------------------------------------------
# Plain-language findings
# --------------------------------------------------------------------------
def _fmt(x, digits: int = 4) -> str:
    return f"{x:.{digits}g}"


def _setting_text(opt: dict, factors: list[Factor]) -> str:
    s = opt["setting"]
    main = s[s["Counted in prediction"] == "yes"]
    weak = s[s["Counted in prediction"] != "yes"]
    text = "; ".join(f"{r['Factor']} = {r['Best level']}" for _, r in main.iterrows())
    if len(weak):
        text += " (factors with little effect: " + "; ".join(f"{r['Factor']} = {r['Best level']}" for _, r in weak.iterrows()) + ")"
    return text


def strip_markup(text: str) -> str:
    text = re.sub(r":[a-z-]+\[(.*?)\]", r"\1", text)
    return text.replace("**", "")


def build_findings(design: pd.DataFrame, factors: list[Factor], analysis: dict, units: dict,
                   goals: dict, reference: str | None = None, array_name: str | None = None) -> list[dict]:
    """List of {kind, section, title, text}; kind is success / info / warning."""
    out: list[dict] = []
    diag = analysis["diagnostics"]
    u = lambda k: ulabel(k, units)  # noqa: E731
    names = ", ".join(f"{f.name} ({len(f.levels)} levels)" if f.kind == "cat" else f"{f.name} (straight line)" for f in factors)
    where = f" It follows the {array_name} array." if array_name else ""
    resid = diag["error_df"]
    out.append({"kind": "info", "section": "design", "title": "Your design",
                "text": (f":blue[**{diag['runs']} runs**] and :blue[**{len(factors)} factors**]: {names}.{where} "
                         f"The factors use {diag['params'] - 1} degrees of freedom and leave :blue[**{resid}**] for error"
                         + (", so the design is saturated and the weakest factors are pooled to form F values." if resid == 0 else
                            "." if resid >= 2 else ", which makes F values weak evidence."))})
    if not diag["orthogonal"]:
        a, b = diag["pair"]
        out.append({"kind": "warning", "section": "design", "title": "The design is not orthogonal",
                    "text": (f"{a} and {b} are linked (association {diag['max_assoc']:.2f}). Factor shares overlap, so they "
                             "need not add up to 100% and a factor can borrow credit from another. "
                             "Use the ranking as a guide and confirm the leaders with extra runs.")})
    if diag["aliased"]:
        out.append({"kind": "warning", "section": "design", "title": "Some factors are confounded",
                    "text": (", ".join(diag["aliased"]) + " cannot be told apart from other factors in this design. "
                             "Add runs or change the array.")})
    if diag["unbalanced"]:
        out.append({"kind": "info", "section": "design", "title": "Unequal level counts",
                    "text": ("Levels of " + ", ".join(diag["unbalanced"]) + " do not appear equally often (dummy levels or a hand-made "
                             "table). Level means are still valid, but the ANOVA uses the drop-one method.")})

    for name, per in analysis["comparison"].items():
        for k, m in per.items():
            rating = agreement_rating(m["nrmse_pct"])
            kind = {"Good": "success", "Moderate": "info", "Poor": "warning"}.get(rating, "info")
            unit = units.get(k, "")
            direction = "higher" if m["bias"] > 0 else "lower"
            pct = m["bias_pct_of_ref_mean"]
            pct_txt = f" ({pct:+.1f}% of the {reference} mean)" if pct is not None else ""
            nr = "n/a" if m["nrmse_pct"] is None else f"{m['nrmse_pct']:.1f}%"
            colour = {"Good": "green", "Moderate": "orange", "Poor": "red"}.get(rating, "gray")
            text = (f"{name} reads on average :orange[**{_fmt(abs(m['bias']))}{(' ' + unit) if unit else ''} {direction}**] than "
                    f"{reference}{pct_txt}. NRMSE is :blue[**{nr}**] of the {reference} range, a :{colour}[**{rating.lower()}**] "
                    f"match by the 5% / 15% rule of thumb ({m['n']} paired runs).")
            if m["rho"] is not None and m["r"] is not None:
                if m["rho"] >= 0.9 and rating != "Good":
                    text += (f" Rank correlation is high (ρ = {m['rho']:.2f}), so both sources order the runs almost the same way. "
                             "Either one can pick the better settings, but their absolute levels differ.")
                elif m["rho"] < 0.7:
                    text += (f" Rank correlation is low (ρ = {m['rho']:.2f}): the sources disagree about which runs are high "
                             "and which are low, not only about the level.")
            if m["slope"] is not None and abs(m["slope"] - 1) > 0.1 and m["r"] is not None and m["r"] > 0.9:
                text += f" The fitted slope of {name} against {reference} is {m['slope']:.2f}, so the gap grows with the response."
            out.append({"kind": kind, "section": "compare", "output": k, "title": f"{u(k)}: {name} vs {reference}", "text": text})

    seen = list(dict.fromkeys(k for per in analysis["anova"].values() for k in per))
    for k in seen:
        leaders = {}
        for name, per in analysis["anova"].items():
            if k not in per:
                continue
            table, meta = per[k]
            top = table.iloc[0]
            second = table.iloc[1] if len(table) > 1 else None
            leaders[name] = top["Factor"]
            text = (f"In {name}, :violet[**{top['Factor']}**] explains :blue[**{top['Contribution (%)']:.1f}%**] of the variation"
                    + (f", followed by {second['Factor']} ({second['Contribution (%)']:.1f}%)." if second is not None else ".")
                    + f" The factors together explain {meta['r2_pct']:.1f}%.")
            if np.isfinite(top["p-value"]):
                text += f" For the leader F = {top['F']:.1f}, p = {top['p-value']:.3g} ({top['Verdict'].lower()})."
            if meta["residual_pct"] > 10:
                text += (f" The remaining :orange[{meta['residual_pct']:.1f}%] comes from interactions, noise or curvature the "
                         "main effects miss, so treat the ranking as a guide.")
            if meta["pooled"]:
                text += " Pooled into error: " + ", ".join(meta["pooled"]) + "."
            out.append({"kind": "info", "section": "drivers", "output": k, "title": f"{u(k)}: what drives it ({name})", "text": text})
        if len(set(leaders.values())) > 1:
            parts = "; ".join(f"{n}: {f}" for n, f in leaders.items())
            out.append({"kind": "warning", "section": "drivers", "output": k,
                        "title": f"{u(k)}: datasets disagree on the leading factor",
                        "text": f"{parts}. Check that the sources use the same boundary conditions and the same definition of the output."})

    for k in seen:
        sets = {n: per[k] for n, per in analysis["optimum"].items() if k in per}
        for name, opt in sets.items():
            gtxt = goal_text(opt["goal"])
            unit = units.get(k, "")
            tail = (f" That combination is :green[**run {opt['tested_run']}**], which you already ran."
                    if opt["tested_run"] is not None else
                    f" That combination is not among your runs; the best tested run is run {opt['best_run']} "
                    f"({_fmt(opt['best_value'])}{(' ' + unit) if unit else ''}). Confirm the prediction with one extra run.")
            out.append({"kind": "success", "section": "optimum", "output": k, "title": f"{u(k)}: best settings ({name})",
                        "text": (f"In {name}, for {k} ({gtxt}), the level means point to: {_setting_text(opt, factors)}. "
                                 f"The additive prediction is :green[**{_fmt(opt['predicted'])}{(' ' + unit) if unit else ''}**]."
                                 + tail)})

    gra_runs = {}
    for name, g in analysis["gra"].items():
        gra_runs[name] = g["best_run"]
        row = design.loc[design[RUN] == g["best_run"]].iloc[0]
        lead = g["table"].iloc[0]
        goals_txt = ", ".join(f"{k} ({goal_text(v)})" for k, v in g["goals"].items())
        setting = "; ".join(f"{f.name} = {fmt_level(row[f.name])}{(' ' + f.unit) if f.unit else ''}" for f in factors)
        out.append({"kind": "success", "section": "gra", "title": f"Best run by combined GRA grade ({name})",
                    "text": (f":green[**Run {g['best_run']}**] scores highest (grade {g['best_grade']:.3f}) for {goals_txt}. "
                             f"The factor that moves the grade most is :violet[**{lead['Factor']}**] ({lead['Contribution (%)']:.1f}%). "
                             f"Settings of that run: {setting}. Level means of the grade suggest: {_setting_text(g['optimum'], factors)}.")})
    if len(set(gra_runs.values())) == 1 and len(gra_runs) > 1:
        out.append({"kind": "success", "section": "gra", "title": "All datasets pick the same best run",
                    "text": f"{' and '.join(gra_runs)} agree on :green[**run {next(iter(gra_runs.values()))}**]. That is a strong sign the choice does not depend on the source."})
    elif len(set(gra_runs.values())) > 1:
        out.append({"kind": "warning", "section": "gra", "title": "Datasets pick different best runs",
                    "text": ("; ".join(f"{n}: run {r}" for n, r in gra_runs.items()) +
                             ". Look at the GRA ranking tables: if the top few grades are close, the sources are not contradicting each other.")})

    for note in analysis["notes"]:
        out.append({"kind": "warning", "section": "caution", "title": "Skipped step", "text": note})
    out.append({"kind": "info", "section": "caution", "title": "How to read these results",
                "text": ("Contribution percentages describe main effects over the levels you tested. They hold for your ranges only and say "
                         "nothing about interactions. The predicted best setting assumes effects add up, so confirm it with a run. "
                         "F and p values are screening aids: CFD is deterministic and test replicates are not separated from "
                         "model error unless you ran repeats.")})
    return out


# --------------------------------------------------------------------------
# Excel export
# --------------------------------------------------------------------------
def _sheet(name: str, used: set) -> str:
    base = re.sub(r"[\[\]:*?/\\]", "_", name)[:31]
    cand, i = base, 2
    while cand in used:
        cand = f"{base[:28]}_{i}"
        i += 1
    used.add(cand)
    return cand


def export_workbook(design: pd.DataFrame, factors: list[Factor], datasets: dict, analysis: dict,
                    findings: list[dict], units: dict) -> bytes:
    buf = io.BytesIO()
    used: set = set()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"Finding": [f["title"] for f in findings],
                      "Detail": [strip_markup(f["text"]) for f in findings]}).to_excel(
            xw, sheet_name=_sheet("Findings", used), index=False)
        dsheet = design.copy()
        dsheet.columns = [RUN] + [f.label for f in factors]
        dsheet.to_excel(xw, sheet_name=_sheet("Design", used), index=False)
        for name, df in datasets.items():
            d = df.rename(columns={c: ulabel(c, units) for c in df.columns if c != RUN})
            d.to_excel(xw, sheet_name=_sheet(f"Data {name}", used), index=False)
        rows = []
        for name, per in analysis["comparison"].items():
            for k, m in per.items():
                rows.append({"Dataset": name, "Output": k, "Unit": units.get(k, ""), "Paired runs": m["n"],
                             "Mean error": m["bias"], "MAE": m["mae"], "RMSE": m["rmse"],
                             "Max abs error": m["max_abs_error"], "NRMSE (%)": m["nrmse_pct"],
                             "MAPE (%)": m["mape_pct"], "Pearson r": m["r"], "Spearman rho": m["rho"],
                             "Slope": m["slope"], "Agreement": agreement_rating(m["nrmse_pct"])})
        if rows:
            pd.DataFrame(rows).to_excel(xw, sheet_name=_sheet("Comparison", used), index=False)
        contrib, lvl, best = [], [], []
        for name, per in analysis["anova"].items():
            for k, (table, meta) in per.items():
                t = table.copy()
                t.insert(0, "Output", k)
                t.insert(0, "Dataset", name)
                t["R2 (%)"], t["Residual (%)"], t["Error df used"] = meta["r2_pct"], meta["residual_pct"], meta["df_error_used"]
                contrib.append(t)
                L = analysis["levels"][name][k].copy()
                L.insert(0, "Output", k)
                L.insert(0, "Dataset", name)
                lvl.append(L)
                s = analysis["optimum"][name][k]["setting"].copy()
                s.insert(0, "Output", k)
                s.insert(0, "Dataset", name)
                s["Predicted"] = analysis["optimum"][name][k]["predicted"]
                best.append(s)
        for title, parts in (("Factor contributions", contrib), ("Level means", lvl), ("Best settings", best)):
            if parts:
                pd.concat(parts).to_excel(xw, sheet_name=_sheet(title, used), index=False)
        gra_rows = []
        for name, g in analysis["gra"].items():
            t = g["ranking"].copy()
            t.insert(0, "Dataset", name)
            gra_rows.append(t)
        if gra_rows:
            pd.concat(gra_rows).to_excel(xw, sheet_name=_sheet("GRA ranking", used), index=False)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Demo data
# --------------------------------------------------------------------------
DEMO_OUTPUTS = [("Heat transfer rate", "W", 4200.0, 0.30),
                ("Coolant outlet temperature", "K", 349.0, 0.05),
                ("Air pressure drop", "Pa", 85.0, 0.45)]


def _position(f: Factor, series) -> np.ndarray:
    """Place a factor's values on -1 .. +1."""
    if f.kind == "lin":
        x = pd.to_numeric(pd.Series(series)).to_numpy(float)
        return np.zeros(len(x)) if np.ptp(x) == 0 else 2 * (x - x.min()) / np.ptp(x) - 1
    idx = level_index(f, series)
    return 2 * idx / max(1, len(f.levels) - 1) - 1


def demo_results(design: pd.DataFrame, factors: list[Factor], kind: str = "CFD", seed: int = 7) -> pd.DataFrame:
    """Made-up results for trying the app. The trend is shared; CFD is smooth, 'Experiment' is offset and noisier."""
    rng0 = np.random.default_rng(11)
    k = len(factors)
    z = np.column_stack([_position(f, design[f.name]) for f in factors])
    scale = 1.0 / (1.0 + np.arange(k)) ** 1.1
    def coefficients():
        return rng0.permutation(scale) * rng0.choice([-1.0, 1.0], k)
    c1 = coefficients()
    coef = [c1, -0.8 * c1 + 0.3 * coefficients(), coefficients()]
    curve = rng0.normal(0, 0.3, k)
    rng = np.random.default_rng(seed)
    out = {RUN: design[RUN].to_numpy()}
    for (name, unit, base, amp), c in zip(DEMO_OUTPUTS, coef):
        trend = (z @ c) / np.abs(c).sum() + 0.15 * ((z ** 2 - 1 / 3) @ curve) / max(1, np.abs(curve).sum())
        y = base * (1 + amp * trend)
        if kind.lower().startswith("exp"):
            y = y + 0.15 * amp * base + rng.normal(0, 0.06 * amp * base, len(y))
        else:
            y = y + rng.normal(0, 0.02 * amp * base, len(y))
        out[f"{name} ({unit})"] = y
    return pd.DataFrame(out)
