"""DOE Results Analyzer (Streamlit).

No physics model inside. You define the design of experiments (factors and levels), run the
cases in CFD or on a test rig, enter the results, and the app does the statistics: comparison
between sources, factor contribution (ANOVA), best settings and grey relational analysis (GRA).

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import doe_core as dc

DEFAULT_NAMES = ["CFD", "Experiment", "Dataset 3"]
MAX_DATASETS = 3
RUN = dc.RUN
DEFAULT_OUTPUTS = "Heat transfer rate (W), Coolant outlet temperature (K), Air pressure drop (Pa)"
EXAMPLE_FACTORS = [
    {"Factor": "Air inlet velocity", "Unit": "m/s", "Levels": "2, 4, 6, 8, 10"},
    {"Factor": "Water inlet temperature", "Unit": "K", "Levels": "333.15, 343.15, 353.15, 363.15, 373.15"},
    {"Factor": "Fin pitch", "Unit": "mm", "Levels": "2, 3, 4, 5, 6"},
    {"Factor": "Tube pitch", "Unit": "mm", "Levels": "25, 30, 35, 40, 45"},
]
GOAL_LABELS = {"Lower is better": "lower", "Higher is better": "higher", "Closest to a target": "target"}

st.set_page_config(
    page_title="DOE Results Analyzer",
    page_icon=":material/grid_on:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 4.2rem; max-width: 1360px;}
[data-testid="stMetricValue"] {font-size: 1.5rem; line-height: 1.25;}
[data-testid="stMetricValue"] > div {white-space: normal; overflow: visible; text-overflow: clip;}
.doe-hero h1 {font-size: 2rem; margin: 0; padding: 0; letter-spacing: -0.01em;}
.doe-hero p {margin: .35rem 0 0; opacity: .78; font-size: 1.02rem; max-width: 62rem;}
.doe-steps {display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; margin: 1rem 0 1.1rem;}
.doe-step {border: 1px solid rgba(128,128,128,.30); border-radius: .6rem; padding: .6rem .8rem;
           font-size: .92rem; line-height: 1.35;}
.doe-step b {display: block; font-size: .74rem; text-transform: uppercase; letter-spacing: .07em;
             opacity: .62; margin-bottom: .15rem;}
@media (max-width: 800px) {.doe-steps {grid-template-columns: 1fr 1fr;}}
</style>
<div class="doe-hero">
  <h1>DOE Results Analyzer</h1>
  <p>Set up a design of experiments, enter the CFD and test results you get for each run, and see which
  factors matter, how the sources compare and which settings work best. One output or several.</p>
</div>
<div class="doe-steps">
  <div class="doe-step"><b>Step 1 &middot; Design</b>List your factors and levels. The app builds the run list.</div>
  <div class="doe-step"><b>Step 2 &middot; Results</b>Type in or upload the result of every run, per source.</div>
  <div class="doe-step"><b>Step 3 &middot; Choose</b>Pick outputs, goals and the reference in the sidebar.</div>
  <div class="doe-step"><b>Step 4 &middot; Analysis</b>Read the summary, drivers, best settings and best run.</div>
</div>
""",
    unsafe_allow_html=True,
)

if "fac_ver" not in st.session_state:
    st.session_state.fac_ver = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sig(*parts) -> str:
    return hashlib.md5(repr(parts).encode()).hexdigest()[:8]


def read_table(file) -> pd.DataFrame:
    file.seek(0)
    if file.name.lower().endswith((".xlsx", ".xlsm")):
        return pd.read_excel(file)
    return pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig", encoding_errors="replace")


def read_pasted(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")


def to_xlsx(df: pd.DataFrame, sheet: str) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name=sheet, index=False)
    return buf.getvalue()


XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def num(x, digits: int = 4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}g}"


# ---------------------------------------------------------------------------
# Chart styling (fixed categorical order by dataset, validated for 4 series on light and dark)
# ---------------------------------------------------------------------------
def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def ink() -> dict:
    if theme_mode() == "dark":
        return dict(bg="#0e1117", text="#c3c2b7", muted="#898781", grid="#2c2c2a", base="#383835",
                    gray="#52514e", series=["#3987e5", "#d95926", "#199e70", "#c98500"])
    return dict(bg="#ffffff", text="#52514e", muted="#898781", grid="#e1e0d9", base="#c3c2b7",
                gray="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])


def series_color(name: str, order: list[str]) -> str:
    c = ink()["series"]
    return c[order.index(name) % len(c)] if name in order else c[0]


def style(fig: go.Figure, height: int, legend: bool = True) -> go.Figure:
    c = ink()
    fig.update_layout(
        height=height, margin=dict(l=8, r=12, t=34 if legend else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=13, color=c["text"]),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=12),
    )
    for upd in (fig.update_xaxes, fig.update_yaxes):
        upd(gridcolor=c["grid"], gridwidth=1, zeroline=False, linecolor=c["base"],
            tickfont=dict(color=c["muted"]), automargin=True)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def parity_fig(ref, other, runs, xtitle: str, ytitle: str, name: str, color: str) -> go.Figure:
    c = ink()
    ok = np.isfinite(ref) & np.isfinite(other)
    ref, other, runs = ref[ok], other[ok], runs[ok]
    lo, hi = float(min(ref.min(), other.min())), float(max(ref.max(), other.max()))
    pad = 0.05 * ((hi - lo) or 1.0)
    rng = [lo - pad, hi + pad]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rng, y=rng, mode="lines", name="Perfect agreement", hoverinfo="skip",
                             line=dict(color=c["muted"], width=1.5)))
    fig.add_trace(go.Scatter(
        x=ref, y=other, mode="markers", name=name, customdata=runs,
        marker=dict(size=9, color=color, line=dict(width=2, color=c["bg"])),
        hovertemplate=f"Run %{{customdata}}<br>{xtitle} %{{x:.4g}}<br>{ytitle} %{{y:.4g}}<extra></extra>"))
    fig.update_xaxes(title=xtitle, range=rng)
    fig.update_yaxes(title=ytitle, range=rng)
    return style(fig, 420)


def runs_fig(series: dict[str, np.ndarray], runs, ytitle: str, order: list[str]) -> go.Figure:
    c = ink()
    fig = go.Figure()
    for name, vals in series.items():
        fig.add_trace(go.Scatter(
            x=runs, y=vals, mode="markers", name=name,
            marker=dict(size=8, color=series_color(name, order), line=dict(width=2, color=c["bg"])),
            hovertemplate=f"Run %{{x}}<br>{name}: %{{y:.4g}}<extra></extra>"))
    fig.update_xaxes(title="Run")
    fig.update_yaxes(title=ytitle)
    return style(fig, 360, legend=len(series) > 1)


def error_fig(errors: dict[str, np.ndarray], runs, unit: str, ref: str, order: list[str]) -> go.Figure:
    c = ink()
    names = list(errors)
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=[f"{n} minus {ref}" for n in names])
    for i, name in enumerate(names, start=1):
        fig.add_trace(go.Bar(
            x=runs, y=errors[name], name=name, showlegend=False,
            marker=dict(color=series_color(name, order), line=dict(width=1, color=c["bg"])),
            hovertemplate=f"Run %{{x}}<br>Error %{{y:.4g}} {unit}<extra></extra>"), row=i, col=1)
        fig.update_yaxes(title=unit or None, row=i, col=1, zeroline=True, zerolinecolor=c["base"])
    fig.update_xaxes(title="Run", row=len(names), col=1)
    fig.update_layout(bargap=0.15)
    for a in fig.layout.annotations:
        a.update(font=dict(size=12, color=c["text"]), x=0, xanchor="left")
    style(fig, 120 + 160 * len(names), legend=False)
    fig.update_layout(margin=dict(l=8, r=12, t=44, b=8))
    return fig


def contrib_fig(tables: dict[str, pd.DataFrame], order: list[str], xtitle: str) -> go.Figure:
    c = ink()
    first = next(iter(tables.values()))
    avg = {f: float(np.mean([t.set_index("Factor").loc[f, "Contribution (%)"] for t in tables.values()]))
           for f in first["Factor"]}
    factor_order = sorted(avg, key=avg.get)  # smallest first, so the largest sits on top
    fig = go.Figure()
    for name in reversed(list(tables)):
        t = tables[name].set_index("Factor").loc[factor_order]
        vals = t["Contribution (%)"].to_numpy()
        fig.add_trace(go.Bar(
            y=factor_order, x=vals, orientation="h", name=name,
            marker=dict(color=series_color(name, order), line=dict(width=1, color=c["bg"])),
            text=[f"{v:.1f}%" if v >= 5 else "" for v in vals], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>" + name + ": %{x:.2f}%<extra></extra>"))
    n = len(tables)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06, legend_traceorder="reversed")
    fig.update_xaxes(title=xtitle, ticksuffix="%", rangemode="tozero")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=c["text"]))
    return style(fig, 90 + len(factor_order) * (n * 19 + 20), legend=n > 1)


def main_effects_fig(levels: dict[str, pd.DataFrame], cat: list[dc.Factor], order: list[str],
                     same_scale: bool, ytitle: str) -> go.Figure:
    c = ink()
    ncol = min(3, len(cat))
    nrow = int(np.ceil(len(cat) / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, vertical_spacing=0.2 / nrow * 2, horizontal_spacing=0.06,
                        subplot_titles=[f.label for f in cat])
    for idx, f in enumerate(cat):
        r, col = idx // ncol + 1, idx % ncol + 1
        for name, lv in levels.items():
            sub = lv[lv["Factor"] == f.name]
            fig.add_trace(go.Scatter(
                x=sub["Level #"], y=sub["Mean"], mode="lines+markers", name=name, showlegend=idx == 0,
                legendgroup=name, customdata=sub["Level"],
                line=dict(width=2, color=series_color(name, order)),
                marker=dict(size=8, color=series_color(name, order), line=dict(width=2, color=c["bg"])),
                hovertemplate=f"{f.name} = %{{customdata}} {f.unit}<br>{name}: %{{y:.4g}}<extra></extra>"),
                row=r, col=col)
        fig.update_xaxes(tickvals=list(range(1, len(f.levels) + 1)), ticktext=f.level_labels(), row=r, col=col)
        if same_scale:
            fig.update_yaxes(matches="y", row=r, col=col)
    fig.update_yaxes(title=ytitle, col=1)
    for a in fig.layout.annotations:
        a.update(font=dict(size=12, color=c["text"]))
    style(fig, 150 + 250 * nrow)
    fig.update_layout(margin=dict(l=8, r=12, t=84, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0 + 0.1 / nrow, xanchor="left", x=0))
    return fig


def trend_fig(design: pd.DataFrame, frames: dict[str, pd.DataFrame], key: str, lin: list[dc.Factor],
              order: list[str], ytitle: str) -> go.Figure:
    c = ink()
    ncol = min(3, len(lin))
    nrow = int(np.ceil(len(lin) / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, vertical_spacing=0.2 / nrow * 2, horizontal_spacing=0.06,
                        subplot_titles=[f.label for f in lin])
    for idx, f in enumerate(lin):
        r, col = idx // ncol + 1, idx % ncol + 1
        x = pd.to_numeric(design[f.name]).to_numpy(float)
        for name, df in frames.items():
            y = df[key].to_numpy(float)
            ok = np.isfinite(y)
            fig.add_trace(go.Scatter(
                x=x[ok], y=y[ok], mode="markers", name=name, showlegend=idx == 0, legendgroup=name,
                marker=dict(size=7, color=series_color(name, order), line=dict(width=1.5, color=c["bg"])),
                hovertemplate=f"{f.name} = %{{x:.5g}}<br>{name}: %{{y:.4g}}<extra></extra>"), row=r, col=col)
            if ok.sum() > 2 and np.ptp(x[ok]) > 0:
                slope, icpt = np.polyfit(x[ok], y[ok], 1)
                xs = np.array([x[ok].min(), x[ok].max()])
                fig.add_trace(go.Scatter(x=xs, y=slope * xs + icpt, mode="lines", showlegend=False, hoverinfo="skip",
                                         legendgroup=name, line=dict(width=1.5, color=series_color(name, order))),
                              row=r, col=col)
    fig.update_yaxes(title=ytitle, col=1)
    for a in fig.layout.annotations:
        a.update(font=dict(size=12, color=c["text"]))
    style(fig, 150 + 250 * nrow)
    fig.update_layout(margin=dict(l=8, r=12, t=84, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0 + 0.1 / nrow, xanchor="left", x=0))
    return fig


def gra_rank_fig(ranking: pd.DataFrame, color: str, top: int = 10) -> go.Figure:
    c = ink()
    t = ranking.head(top).iloc[::-1]
    colors = [c["gray"]] * (len(t) - 1) + [color]
    fig = go.Figure(go.Bar(
        y=[f"Run {int(r)}" for r in t[RUN]], x=t["GRA grade"], orientation="h",
        marker=dict(color=colors, line=dict(width=1, color=c["bg"])),
        text=[f"{v:.3f}" for v in t["GRA grade"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Grade %{x:.4f}<extra></extra>"))
    fig.update_layout(bargap=0.35)
    fig.update_xaxes(title="GRA grade (higher is better)", range=[0, max(1.0, float(t["GRA grade"].max()) * 1.12)])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=c["text"]))
    return style(fig, 70 + 30 * len(t), legend=False)


tab_design, tab_data, tab_res = st.tabs([
    ":material/grid_on: 1 · Design",
    ":material/edit_note: 2 · Results data",
    ":material/insights: 3 · Analysis",
])

# ===========================================================================
# TAB 1 - DESIGN
# ===========================================================================
design: pd.DataFrame | None = None
factors: list[dc.Factor] = []
design_meta: dict = {}
with tab_design:
    st.markdown("#### Define the design of experiments")
    mode = st.radio("How do you want to set up the runs?",
                    ["Build it here from factors and levels", "I already have a design table"],
                    horizontal=True, key="design_mode")

    # ------------------------------------------------------------ build from factors
    if mode.startswith("Build"):
        st.markdown(
            "Give each factor a name, a unit and its **levels** (the values you will test), separated by commas. "
            "The app picks a :blue[Taguchi orthogonal array] that covers the levels in the fewest runs, so you run "
            "CFD or tests only for the rows it gives you. Text levels such as `low, high` work too."
        )
        base = pd.DataFrame(EXAMPLE_FACTORS)
        edited = st.data_editor(
            base, num_rows="dynamic", hide_index=True, width="stretch", key=f"factors_{st.session_state.fac_ver}",
            column_config={
                "Factor": st.column_config.TextColumn("Factor", required=True, width="medium"),
                "Unit": st.column_config.TextColumn("Unit", width="small"),
                "Levels": st.column_config.TextColumn("Levels (comma separated)", required=True, width="large"),
            })
        b1, b2 = st.columns([1, 3])
        if b1.button("Reset to the example", icon=":material/restart_alt:"):
            st.session_state.fac_ver += 1
            st.session_state.pop("oa_choice", None)
            st.rerun()
        b2.caption("Add rows at the bottom of the table. Each factor needs 2 to 8 levels. The example is a crossflow "
                   "radiator study; replace it with your own factors.")

        factors, errs = dc.factors_from_table(edited)
        if errs:
            st.error("Fix the factor table first:\n\n" + "\n".join(f"- {e}" for e in errs), icon=":material/error:")
            factors = []
        if factors:
            counts = [len(f.levels) for f in factors]
            need = 1 + sum(n - 1 for n in counts)
            options: dict[str, str] = {}
            for a in dc.compatible_arrays(counts):
                label = f"{a['name']} · {a['runs']} runs · {a['error_df']} spare degrees of freedom"
                if a["dummy"]:
                    label += f" · {a['dummy']} factor(s) use dummy levels"
                options[a["key"]] = label
            full_n = dc.full_factorial_runs(factors)
            if full_n <= 300:
                options["FULL"] = f"Full factorial · {full_n} runs · {full_n - need} spare degrees of freedom"
            if st.session_state.get("oa_choice") not in options:
                st.session_state.pop("oa_choice", None)
            if not options:
                st.error("No standard array in the catalog carries these factors, and a full factorial would need "
                         f"{full_n:,} runs. Use fewer factors or levels, or switch to \"I already have a design table\".",
                         icon=":material/error:")
                factors = []
            else:
                choice = st.selectbox(
                    "Design type", list(options), format_func=options.get, key="oa_choice",
                    help="An orthogonal array tests a balanced subset of all combinations, so each factor's effect can be "
                         "separated from the others. A full factorial tests every combination. Spare degrees of freedom "
                         "are what is left after the factors take their share; they give the error estimate.")
                design, design_meta = (dc.full_factorial(factors) if choice == "FULL" else dc.build_design(factors, choice))
                with st.expander("Which design should I pick?"):
                    st.markdown(
                        "- The smallest array is the cheapest in CFD or test time. It is the first in the list.\n"
                        "- **Spare degrees of freedom** below 2 leave little or nothing to judge noise with. The app pools the "
                        "weakest factors into error to still give F values, but the factor ranking is the safer result.\n"
                        "- A **dummy level** repeats a level of a factor that has fewer levels than its array column. "
                        "The data stay valid, with some levels tested more often than others.\n"
                        "- A **full factorial** also lets you check interactions later, at the cost of many more runs."
                    )

    # ------------------------------------------------------------ own design table
    else:
        st.markdown(
            "Upload or paste the table of runs you already use: **one row per run, one column per factor**. A `Run` column is "
            "optional. Put the unit in brackets in the header, for example `Fin pitch (mm)`."
        )
        example = dc.build_design(dc.factors_from_table(pd.DataFrame(EXAMPLE_FACTORS))[0], "L25")[0]
        src = st.radio("Where is the table?", ["Upload a file", "Paste from Excel"], horizontal=True, key="design_src")
        raw_design = None
        try:
            if src == "Upload a file":
                up = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xlsm"], key="design_file")
                raw_design = read_table(up) if up is not None else None
            else:
                txt = st.text_area("Paste the table with its header row", height=150, key="design_text")
                raw_design = read_pasted(txt) if txt.strip() else None
        except Exception as exc:
            st.error(f"Could not read the table: {exc}", icon=":material/error:")
        st.download_button("Download an example design (CSV)", example.to_csv(index=False), file_name="example_design.csv",
                           mime="text/csv", icon=":material/download:")
        if raw_design is not None:
            prof = dc.profile_columns(raw_design)
            if prof.empty:
                st.error("The table has no usable columns.", icon=":material/error:")
            else:
                chosen = st.multiselect(
                    "Factor columns", list(prof["Column"]), default=list(prof["Column"]), key="design_cols",
                    help="Every column is read as a factor. Remove any column that holds results or notes.")
                numeric_many = [r.Column for r in prof.itertuples() if r.Numeric and r.Distinct > 2 and r.Column in chosen]
                lin_default = [c for c in numeric_many if prof.set_index("Column").loc[c, "Distinct"] > dc.MAX_CAT_LEVELS]
                lin_cols = st.multiselect(
                    "Treat as a straight-line trend instead of separate levels", numeric_many, default=lin_default,
                    key="design_lin",
                    help="Use this for factors with many different values (a space-filling or random design). The app then "
                         "fits one slope per factor instead of comparing level means.")
                try:
                    kinds = {c: ("lin" if c in lin_cols else "cat") for c in numeric_many}
                    design, factors, notes = dc.parse_design(raw_design, chosen, kinds)
                    design_meta = {"array": "your own table", "key": "OWN", "runs": len(design), "dummy": [], "spare": 0}
                    for n in notes:
                        st.info(n, icon=":material/info:")
                except ValueError as exc:
                    st.error(str(exc), icon=":material/error:")
                    design, factors = None, []

    # ------------------------------------------------------------ design summary
    if design is not None and factors:
        diag = dc.design_diagnostics(design, factors)
        st.markdown("#### Your design")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Runs", f"{diag['runs']}", border=True, help="Each run is one CFD case or one test.")
        m2.metric("Factors", f"{len(factors)}", border=True)
        m3.metric("Degrees of freedom used", f"{diag['params'] - 1}", border=True,
                  help="Each factor with k levels uses k - 1; a straight-line factor uses 1.")
        m4.metric("Left for error", f"{diag['error_df']}", border=True,
                  help="Runs minus 1 minus the degrees of freedom used. Zero means a saturated design.")
        if diag["aliased"]:
            st.error("These factors cannot be told apart from the others in this design: " + ", ".join(diag["aliased"])
                     + ". Add runs or change the design.", icon=":material/error:")
        elif not diag["orthogonal"]:
            a, b = diag["pair"]
            st.warning(f"The design is **not orthogonal**: {a} and {b} are linked (association {diag['max_assoc']:.2f}). "
                       "Factor shares will overlap, so read the ranking as a guide.", icon=":material/warning:")
        elif diag["unbalanced"]:
            st.info("Orthogonal, but some levels are tested more often than others (" + ", ".join(diag["unbalanced"])
                    + "). That comes from dummy levels or from your table.", icon=":material/info:")
        else:
            st.success("Balanced and orthogonal: every level of every factor appears equally often, and the factors are "
                       "independent of each other.", icon=":material/check_circle:")
        if diag["error_df"] == 0:
            st.info("Saturated design: no degrees of freedom are left for error. Contribution percentages still work; "
                    "the app pools the weakest factors to form F values.", icon=":material/info:")
        with st.expander("Factors as the app reads them", expanded=mode.startswith("I already")):
            st.dataframe(pd.DataFrame({
                "Factor": [f.name for f in factors], "Unit": [f.unit for f in factors],
                "Treated as": [f"{len(f.levels)} levels" if f.kind == "cat" else "straight-line trend" for f in factors],
                "Levels or range": [", ".join(f.level_labels()) if f.kind == "cat"
                                    else f"{dc.fmt_level(min(f.levels))} to {dc.fmt_level(max(f.levels))}" for f in factors]}),
                hide_index=True, width="stretch")
        with st.expander(f"Show the {diag['runs']} runs"):
            shown = design.copy()
            shown.columns = [RUN] + [f.label for f in factors]
            st.dataframe(shown, hide_index=True, width="stretch", height=min(420, 38 + 35 * len(shown)))
    elif design is None and mode.startswith("I already"):
        st.caption("Waiting for a design table.")

# ===========================================================================
# TAB 2 - RESULTS DATA
# ===========================================================================
items: dict[str, tuple[pd.DataFrame, dict]] = {}
with tab_data:
    st.markdown("#### Enter or load the results")
    if design is None:
        st.warning("Define the design in step 1 first. The run list depends on it.", icon=":material/arrow_back:")
    else:
        st.markdown(
            "Provide **one value per run** for each output, from CFD, from the test rig, or both. Name the outputs you will "
            "record, put the unit in brackets, then type the values into the table, upload a file or paste from Excel. "
            "You can leave some runs empty and fill them later."
        )
        out_text = st.text_input("Outputs you will record (comma separated)", value=DEFAULT_OUTPUTS, key="outputs_text",
                                 help="The same output in two datasets is matched by name, ignoring capitals and spaces.")
        out_names = [o.strip() for o in re.split(r"[,;\n]+", out_text) if o.strip()]
        if len(out_names) != len({dc._norm(o) for o in out_names}):
            st.error("Two outputs have the same name.", icon=":material/error:")
            out_names = []
        elif not out_names:
            st.error("Name at least one output.", icon=":material/error:")

        if out_names:
            sheet = dc.run_sheet(design, factors, out_names)
            d1, d2, d3 = st.columns([1, 1, 2])
            d1.download_button("Run sheet (CSV)", sheet.to_csv(index=False), file_name="doe_run_sheet.csv",
                               mime="text/csv", icon=":material/download:", width="stretch",
                               help="Every run with its factor settings, a suggested test order and empty result columns.")
            d2.download_button("Run sheet (Excel)", to_xlsx(sheet, "Run sheet"), file_name="doe_run_sheet.xlsx",
                               mime=XLSX, icon=":material/table:", width="stretch")
            with d3.expander("Accepted file layout"):
                st.markdown(
                    "- An optional **Run** column with the run numbers of the design. Without it, rows are read in order.\n"
                    "- One column per output, with the unit in brackets, such as `Heat transfer rate (W)`.\n"
                    "- Factor columns and `Suggested order` are ignored, so you can fill in the run sheet and upload it unchanged.\n"
                    "- Missing runs may stay empty. Repeated run numbers are averaged.\n"
                    "- CSV (comma, semicolon or tab separated) or Excel (.xlsx)."
                )

            n_ds = st.segmented_control("How many datasets (sources)?", list(range(1, MAX_DATASETS + 1)), default=1,
                                        key="n_ds")
            n_ds = int(n_ds or 1)
            used_names: set[str] = set()
            for i in range(n_ds):
                with st.container(border=True):
                    h1, h2 = st.columns([1, 2], gap="large")
                    name = h1.text_input("Dataset name", value=DEFAULT_NAMES[i], key=f"ds_name_{i}", max_chars=24).strip()
                    source = h2.radio("Where do the results come from?",
                                      ["Type into a table", "Upload a file", "Paste from Excel", "Demo data"],
                                      horizontal=True, key=f"ds_src_{i}")
                    raw = None
                    typed = False
                    try:
                        if source == "Type into a table":
                            typed = True
                            grid = sheet.drop(columns=["Suggested order"])
                            lock = [RUN] + [f.label for f in factors]
                            key = f"ds_edit_{i}_{sig(grid.columns.tolist(), len(grid), design.to_numpy().tolist())}"
                            cfg = {c: st.column_config.NumberColumn(c, format="%.6g") for c in out_names}
                            st.caption("Click a cell and type a number. Cells you leave empty show None and are skipped.")
                            got = st.data_editor(grid, hide_index=True, width="stretch", disabled=lock, key=key,
                                                 column_config=cfg, height=min(460, 38 + 35 * len(grid)))
                            raw = got[[RUN] + out_names] if got[out_names].notna().any().any() else None
                        elif source == "Upload a file":
                            up = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xlsm"], key=f"ds_file_{i}")
                            raw = read_table(up) if up is not None else None
                        elif source == "Paste from Excel":
                            txt = st.text_area("Paste the table with its header row (copy cells in Excel, paste here)",
                                               height=140, key=f"ds_text_{i}")
                            raw = read_pasted(txt) if txt.strip() else None
                        else:
                            st.warning("These are **synthetic numbers** with a built-in trend, small scatter and an offset "
                                       "for the experiment, so you can try the app. They are not measurements.",
                                       icon=":material/science:")
                            raw = dc.demo_results(design, factors, "CFD" if i == 0 else "Experiment", seed=7 + i)
                    except Exception as exc:  # unreadable file or text
                        st.error(f"Could not read the table: {exc}", icon=":material/error:")
                        raw = None

                    if not name:
                        st.error("Give the dataset a name.", icon=":material/error:")
                    elif name.casefold() in used_names:
                        st.error(f"'{name}' is already used. Pick a different name.", icon=":material/error:")
                    elif raw is not None:
                        try:
                            parsed, units_i, notes = dc.parse_results(raw, design, factors)
                        except ValueError as exc:
                            st.error(str(exc), icon=":material/error:")
                        else:
                            used_names.add(name.casefold())
                            items[name] = (parsed, units_i)
                            cols_found = [c for c in parsed.columns if c != RUN]
                            filled = int(parsed[cols_found].notna().any(axis=1).sum())
                            st.success(f"**{name}**: {filled} of {len(design)} runs have results. "
                                       f"Outputs: {', '.join(cols_found)}.", icon=":material/check_circle:")
                            for n in notes:
                                if typed and n.startswith("The factor columns"):
                                    continue
                                st.info(n, icon=":material/info:")
                            if not typed:
                                with st.expander("Preview"):
                                    st.dataframe(parsed, hide_index=True, width="stretch", height=240)
                    else:
                        st.caption("Waiting for data. Without it, this dataset is left out of the analysis.")

datasets: dict[str, pd.DataFrame] = {}
units: dict[str, str] = {}
if items:
    datasets, units, merge_notes = dc.merge_datasets(items)
    if merge_notes:
        with tab_data:
            for n in merge_notes:
                st.warning(n, icon=":material/warning:")

# ===========================================================================
# SIDEBAR - ANALYSIS CHOICES
# ===========================================================================
selected: list[str] = []
detailed: list[str] = []
reference: str | None = None
goals: dict = {}
gra_cfg = None
zeta = 0.5
with st.sidebar:
    st.markdown("### Choose the analysis")
    if design is None:
        st.info("Define the design in step 1 to unlock the choices.", icon=":material/arrow_back:")
    elif not datasets:
        st.info("Add results in step 2 to unlock the choices.", icon=":material/arrow_back:")
    else:
        options = list(dict.fromkeys(k for d in datasets.values() for k in d.columns if k != RUN))
        selected = st.multiselect("Outputs to analyse", options, default=options,
                                  format_func=lambda k: dc.ulabel(k, units),
                                  help="Every chart, table and finding uses only the outputs you pick here. "
                                       "Pick one for a single-output study or several for a multi-output study.")
        if selected:
            lines = [f"- **{k}**: {', '.join(n for n, d in datasets.items() if k in d.columns)}" for k in selected]
            st.caption("Where each output is available:")
            st.markdown("\n".join(lines))
        else:
            st.warning("Pick at least one output.", icon=":material/warning:")

        if len(datasets) > 1:
            reference = st.selectbox("Reference for the comparison", list(datasets), key="reference",
                                     help="Differences are shown as other dataset minus this one. Use the source you trust more, "
                                          "usually the test data.")
        detailed = st.multiselect("Datasets for factor analysis and GRA", list(datasets), default=list(datasets),
                                  help="The comparison always uses every dataset. This choice controls the factor "
                                       "contribution, best settings and GRA steps.")
        use_gra = st.toggle("Combine outputs with GRA", value=len(selected) >= 2, disabled=len(selected) < 2,
                            help="Grey relational analysis merges several outputs into one grade per run, so you can pick a "
                                 "single best run. It needs at least two outputs.")
        if len(selected) < 2:
            st.caption("GRA needs two or more outputs.")
        weights: dict = {}
        if selected:
            with st.expander("Goals" + (" and weights" if use_gra and len(selected) >= 2 else ""), expanded=True):
                st.caption("Say what a good result looks like for each output. It sets the best settings"
                           + (" and the GRA grade. Weights are relative." if use_gra and len(selected) >= 2 else "."))
                for k in selected:
                    st.markdown(f"**{dc.ulabel(k, units)}**")
                    default = {"lower": 0, "higher": 1}.get(dc.default_goal(k), 1)
                    label = st.selectbox("Goal", list(GOAL_LABELS), index=default, key=f"goal_{k}",
                                         label_visibility="collapsed")
                    goal = GOAL_LABELS[label]
                    if use_gra and len(selected) >= 2:
                        weights[k] = st.number_input("Weight", min_value=0.01, value=1.0, step=0.1, key=f"w_{k}",
                                                     help="Relative weight of this output in the GRA grade.")
                    target = None
                    if goal == "target":
                        pool = np.concatenate([d[k].dropna().to_numpy(float) for d in datasets.values() if k in d.columns])
                        target = st.number_input(f"Target value ({units.get(k) or 'same unit'})", value=float(pool.mean()),
                                                 format="%g", key=f"t_{k}")
                    goals[k] = (goal, target)
                if weights:
                    total = sum(weights.values())
                    st.caption("Normalised weights: " + ", ".join(f"{k} {w / total:.2f}" for k, w in weights.items()))
            if use_gra and len(selected) >= 2:
                gra_cfg = {"keys": selected, "weights": weights}
                with st.expander("Advanced"):
                    zeta = st.number_input("GRA distinguishing coefficient", min_value=0.1, max_value=1.0, value=0.5, step=0.1,
                                           help="0.5 is the usual choice. Smaller values separate the runs more sharply.")

# ===========================================================================
# TAB 3 - ANALYSIS
# ===========================================================================
order = list(datasets)


def render_summary(analysis: dict, findings: list[dict]) -> None:
    n_runs = len(design)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Runs in the design", f"{n_runs}", border=True, help=design_meta.get("array", ""))
    k2.metric("Factors", f"{len(factors)}", border=True)
    k3.metric("Outputs analysed", f"{len(selected)}", border=True)
    k4.metric("Datasets", f"{len(datasets)}", border=True, help=", ".join(datasets))

    for f in (x for x in findings if x["section"] == "design"):
        box = st.warning if f["kind"] == "warning" else st.info
        box(f["text"], icon=":material/warning:" if f["kind"] == "warning" else ":material/info:")

    rows = []
    for ds, per in analysis["anova"].items():
        for k, (table, meta) in per.items():
            top = table.iloc[0]
            opt = analysis["optimum"][ds][k]
            rows.append({"Output": dc.ulabel(k, units), "Dataset": ds, "Leading factor": top["Factor"],
                         "Share (%)": top["Contribution (%)"], "Explained (%)": meta["r2_pct"],
                         "Predicted best": opt["predicted"], "Best tested run": opt["best_run"],
                         "Goal": dc.goal_text(opt["goal"])})
    if rows:
        st.markdown("##### At a glance")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Output": st.column_config.TextColumn(width="medium"),
            "Dataset": st.column_config.TextColumn(width="small"),
            "Share (%)": st.column_config.ProgressColumn("Share (%)", min_value=0, max_value=100, format="%.1f%%",
                                                         help="Part of the variation in this output that the leading factor explains."),
            "Explained (%)": st.column_config.NumberColumn(format="%.1f%%",
                                                           help="Part of the variation all factors explain together."),
            "Predicted best": st.column_config.NumberColumn(format="%.5g"),
        })

    comp = analysis["comparison"]
    if any(comp.values()):
        st.markdown(f"##### How well the sources agree (reference: {reference})")
        badge = {"Good": ":green-badge[:material/check: Good]", "Moderate": ":orange-badge[:material/remove: Moderate]",
                 "Poor": ":red-badge[:material/close: Poor]", "n/a": ":gray-badge[n/a]"}
        md = [f"| Dataset | Output | Mean error (vs {reference}) | NRMSE | Correlation r | Rank correlation ρ | Agreement |",
              "|---|---|---:|---:|---:|---:|---|"]
        for ds, per in comp.items():
            for k, m in per.items():
                nr = "n/a" if m["nrmse_pct"] is None else f"{m['nrmse_pct']:.1f}%"
                md.append(f"| {ds} | {k} | {num(m['bias'])} {units.get(k, '')} | {nr} | {num(m['r'], 3)} | {num(m['rho'], 3)} | "
                          f"{badge[dc.agreement_rating(m['nrmse_pct'])]} |")
        st.markdown("\n".join(md))
        st.caption("NRMSE is the root-mean-square error as a percentage of the reference range: under 5% reads as good, "
                   "5 to 15% as moderate, above 15% as poor. Rank correlation tells whether both sources order the runs the same way.")

    st.markdown("##### Findings by output")
    for k in selected:
        mine = [f for f in findings if f.get("output") == k]
        if not mine:
            continue
        with st.container(border=True):
            st.markdown(f"###### {dc.ulabel(k, units)}")
            for sec, head in (("compare", "Sources"), ("drivers", "Drivers"), ("optimum", "Best settings")):
                part = [x for x in mine if x["section"] == sec]
                if part:
                    st.markdown(f":gray[{head}]")
                    st.markdown("\n\n".join(f["text"] for f in part))

    gra_f = [x for x in findings if x["section"] == "gra"]
    if gra_f:
        st.markdown("##### Best run across all selected outputs")
        for f in gra_f:
            box = st.warning if f["kind"] == "warning" else st.success
            box(f"**{f['title']}**\n\n{f['text']}", icon=":material/emoji_events:" if f["kind"] != "warning" else ":material/compare_arrows:")

    cautions = [x for x in findings if x["section"] == "caution"]
    st.markdown("##### Cautions")
    for f in cautions:
        (st.warning if f["kind"] == "warning" else st.info)(
            f"**{f['title']}**\n\n{f['text']}", icon=":material/warning:" if f["kind"] == "warning" else ":material/info:")

    with st.expander("What do these numbers mean?"):
        st.markdown(
            "- :blue[**Contribution (%)**] is the share of an output's total variation that one factor explains. "
            "The leftover share is interaction, noise or curvature.\n"
            "- :blue[**F and p-value**] test whether a factor's effect stands out from error. With CFD data they only screen "
            "factors, because a deterministic solver has no random noise.\n"
            "- :blue[**Level means**] are the average output at each level of a factor. Steep lines mean strong effects.\n"
            "- :blue[**Delta and rank**] in the response table are the gap between the best and worst level mean and the order of factors by that gap.\n"
            "- :blue[**Predicted best**] adds up the best level of each factor that explains at least 5%. Confirm it with an extra run.\n"
            "- :blue[**GRA grade**] scores each run from 0 to 1 against your goals for all outputs. The highest grade is the best compromise among the tested runs.")


def render_compare(analysis: dict) -> None:
    comp = analysis["comparison"]
    keys = list(dict.fromkeys(k for per in comp.values() for k in per))
    if not keys:
        st.info("Load a second dataset in step 2 that shares an output with the reference to see the comparison.",
                icon=":material/info:")
        return
    key = st.radio("Output to compare", keys, format_func=lambda k: dc.ulabel(k, units), horizontal=True, key="cmp_out")
    unit = units.get(key, "")
    ref = datasets[reference]
    sets = {n: d for n, d in datasets.items() if n != reference and key in d.columns and key in comp.get(n, {})}
    runs = design[RUN].to_numpy()
    rv = ref[key].to_numpy(float)

    st.markdown("##### Error metrics")
    rows = []
    for ds in sets:
        m = comp[ds][key]
        rows.append({"Dataset": ds, "Paired runs": m["n"], "Mean error": m["bias"], "MAE": m["mae"], "RMSE": m["rmse"],
                     "Max |error|": m["max_abs_error"], "NRMSE (%)": m["nrmse_pct"], "Correlation r": m["r"],
                     "Rank correlation ρ": m["rho"], "Slope": m["slope"], "Agreement": dc.agreement_rating(m["nrmse_pct"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.4g")
        for c in ("Mean error", "MAE", "RMSE", "Max |error|", "NRMSE (%)", "Correlation r", "Rank correlation ρ", "Slope")})
    st.caption(f"Errors are the dataset minus {reference}, in {unit or 'the output unit'}. A positive mean error means the "
               f"dataset reads higher than {reference}. A slope away from 1 means the gap changes with the response level.")

    st.markdown(f"##### {reference} against each dataset, run by run")
    reading = (":blue[Points on the line] mean the two agree. :orange[Points above it] mean the dataset reads higher than "
               f"{reference}. A tight cloud off the line is a steady offset; a wide cloud means the sources disagree about "
               "how the runs differ.")
    suffix = f" ({unit})" if unit else ""
    cols = st.columns(max(len(sets), 2))
    if len(sets) > 1:
        st.markdown(reading)
    for col, (ds, d) in zip(cols, sets.items()):
        with col:
            st.markdown(f"**{ds}**")
            show(parity_fig(rv, d[key].to_numpy(float), runs, f"{reference}{suffix}", f"{ds}{suffix}", ds,
                            series_color(ds, order)))
    if len(sets) == 1:
        with cols[1]:
            st.markdown("**How to read this**")
            st.markdown(reading)

    st.markdown("##### All sources by run")
    show(runs_fig({n: d[key].to_numpy(float) for n, d in datasets.items() if key in d.columns}, runs,
                  dc.ulabel(key, units), order))
    st.markdown("##### Error by run")
    show(error_fig({ds: d[key].to_numpy(float) - rv for ds, d in sets.items()}, runs, unit, reference, order))


def render_drivers(analysis: dict) -> None:
    anova = analysis["anova"]
    keys = list(dict.fromkeys(k for per in anova.values() for k in per))
    if not keys:
        st.info("Pick at least one output and one dataset in the sidebar.", icon=":material/info:")
        return
    st.markdown(
        "Each bar is the share of an output's total variation that one factor explains on its own "
        "(:blue[main-effect contribution]). The bars plus the :orange[unexplained share] add to 100% for an orthogonal design.")
    key = st.radio("Output", keys, format_func=lambda k: dc.ulabel(k, units), horizontal=True, key="drv_out")
    per = {n: v[key] for n, v in anova.items() if key in v}
    tables = {n: t for n, (t, _) in per.items()}
    show(contrib_fig(tables, order, f"Contribution to variation in {key.lower()}"))

    cols = st.columns(len(per))
    for col, (n, (t, meta)) in zip(cols, per.items()):
        col.metric(n, f"{meta['r2_pct']:.1f}% explained", border=True,
                   help=f"The factors explain {meta['r2_pct']:.1f}% of the variation. The other {meta['residual_pct']:.1f}% "
                        "comes from interactions, noise or curvature.")
    st.caption("Below about 90% explained, interactions or noise matter and the ranking is only a guide.")
    for n, (t, meta) in per.items():
        for msg in meta["notes"]:
            st.warning(f"{n}: {msg}", icon=":material/warning:")
        if meta["pooled"]:
            st.info(f"{n}: pooled into error to form F values: {', '.join(meta['pooled'])}.", icon=":material/info:")
        if not meta["additive"]:
            st.info(f"{n}: the design is unbalanced or not orthogonal, so factor shares overlap and need not add to 100%.",
                    icon=":material/info:")

    pivot = pd.DataFrame({n: t.set_index("Factor")["Contribution (%)"] for n, t in tables.items()})
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index].reset_index()
    st.dataframe(pivot, hide_index=True, width="stretch", column_config={
        n: st.column_config.ProgressColumn(n, min_value=0, max_value=100, format="%.1f%%") for n in tables})

    cat = [f for f in factors if f.kind == "cat"]
    lin = [f for f in factors if f.kind == "lin"]
    frames = {n: datasets[n] for n in per}
    with st.expander("How the output changes across the levels of each factor", expanded=True):
        if cat:
            st.caption("Average output at each level. A steep line means a strong effect; a flat line means little.")
            same = st.checkbox("Use the same vertical scale for every factor", value=True, key="same_scale")
            show(main_effects_fig({n: analysis["levels"][n][key] for n in per}, cat, order, same, dc.ulabel(key, units)))
        if lin:
            st.caption("Straight-line factors: every run as a point, with the fitted line.")
            show(trend_fig(design, frames, key, lin, order, dc.ulabel(key, units)))
    with st.expander("Response tables (level means, delta and rank)"):
        for n in per:
            st.markdown(f"**{n}**")
            rt = dc.response_table(analysis["levels"][n][key], factors)
            st.dataframe(rt, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.5g") for c in rt.columns if c not in ("Factor", "Rank")})
    with st.expander("Full ANOVA tables"):
        for n, (t, meta) in per.items():
            st.markdown(f"**{n}**  ·  {meta['n']} runs, {meta['df_res']} residual df")
            st.dataframe(t, hide_index=True, width="stretch", column_config={
                "SS": st.column_config.NumberColumn(format="%.5g"), "MS": st.column_config.NumberColumn(format="%.5g"),
                "F": st.column_config.NumberColumn(format="%.4g"), "p-value": st.column_config.NumberColumn(format="%.3g"),
                "Contribution (%)": st.column_config.NumberColumn(format="%.2f"),
                "Effect range": st.column_config.NumberColumn(format="%.5g")})


def render_settings(analysis: dict) -> None:
    optimum = analysis["optimum"]
    keys = list(dict.fromkeys(k for per in optimum.values() for k in per))
    if not keys:
        st.info("Pick at least one output and one dataset in the sidebar.", icon=":material/info:")
        return
    st.markdown(
        "For each output, the best level of every factor comes from the level means and your goal. The prediction adds the "
        "effects of the factors that explain at least 5% of the variation. It assumes effects add up, so run the "
        ":blue[confirmation test] before you rely on it.")
    key = st.radio("Output", keys, format_func=lambda k: dc.ulabel(k, units), horizontal=True, key="opt_out")
    unit = units.get(key, "")
    for ds, per in optimum.items():
        if key not in per:
            continue
        opt = per[key]
        with st.container(border=True):
            st.markdown(f"**{ds}**  ·  {dc.goal_text(opt['goal'])}")
            a, b, c3 = st.columns(3)
            a.metric("Predicted best", f"{num(opt['predicted'], 5)} {unit}".strip(), border=True)
            b.metric("Best tested run", f"Run {opt['best_run']}", border=True,
                     help=f"Value {num(opt['best_value'], 5)} {unit}".strip())
            c3.metric("Best tested value", f"{num(opt['best_value'], 5)} {unit}".strip(), border=True)
            st.dataframe(opt["setting"], hide_index=True, width="stretch", column_config={
                "Mean at that level": st.column_config.NumberColumn(format="%.5g"),
                "Effect vs overall mean": st.column_config.NumberColumn(format="%+.4g")})
            if opt["tested_run"] is not None:
                st.success(f"This combination is run {opt['tested_run']}, which is already in your design.",
                           icon=":material/check_circle:")
            else:
                st.info("This combination is not one of your runs. Run it once (CFD or test) and compare with the "
                        "prediction to confirm.", icon=":material/science:")


def render_gra(analysis: dict) -> None:
    gra = analysis["gra"]
    if gra_cfg is None:
        st.info("Turn on **Combine outputs with GRA** in the sidebar (needs two or more outputs).", icon=":material/info:")
        return
    if not gra:
        st.warning("GRA could not run for the chosen datasets. See the cautions in the summary.", icon=":material/warning:")
        return
    st.markdown(
        "Grey relational analysis turns each output into a 0 to 1 score (1 = the goal you set), merges the scores with your "
        "weights, and gives every run one :blue[grade]. The run with the highest grade is the best compromise among the "
        "tested settings.")
    first = next(iter(gra.values()))
    st.dataframe(pd.DataFrame({
        "Output": [dc.ulabel(k, units) for k in first["keys"]],
        "Goal": [dc.goal_text(first["goals"][k]) for k in first["keys"]],
        "Weight": [first["weights"][k] for k in first["keys"]]}),
        hide_index=True, width="stretch", column_config={"Weight": st.column_config.NumberColumn(format="%.2f")})
    st.caption("If a dataset lacks one of these outputs, that output is left out for that dataset and the other weights are "
               "rescaled to add up to 1. Outputs linked by physics (for example heat rate and outlet temperature) count the "
               "same effect twice, so give them smaller weights.")

    best = {n: g["best_run"] for n, g in gra.items()}
    if len(set(best.values())) > 1:
        st.warning("The datasets pick **different best runs**: " + "; ".join(f"{n} → run {r}" for n, r in best.items())
                   + ". Compare their top-10 lists below before choosing.", icon=":material/compare_arrows:")
    elif len(best) > 1:
        st.success(f"All datasets agree: :green[**run {next(iter(best.values()))}**] is the best run.",
                   icon=":material/check_circle:")

    pick = st.radio("Dataset", list(gra), horizontal=True, key="gra_pick") if len(gra) > 1 else next(iter(gra))
    g = gra[pick]
    run = g["best_run"]
    row_in = design.loc[design[RUN] == run].iloc[0]
    row_out = datasets[pick].loc[datasets[pick][RUN] == run].iloc[0]
    runner = float(g["ranking"].iloc[1]["GRA grade"]) if len(g["ranking"]) > 1 else g["best_grade"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Best run", f"Run {run}", border=True)
    m2.metric("GRA grade", f"{g['best_grade']:.3f}", delta=f"{g['best_grade'] - runner:+.3f} vs second", border=True)
    m3.metric("Main driver of the grade", g["table"].iloc[0]["Factor"], border=True,
              help=f"{g['table'].iloc[0]['Contribution (%)']:.1f}% of the variation in the grade.")

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(f"**Settings of run {run}**")
        st.dataframe(pd.DataFrame({
            "Factor": [f.name for f in factors], "Value": [dc.fmt_level(row_in[f.name]) for f in factors],
            "Unit": [f.unit for f in factors]}), hide_index=True, width="stretch")
    with b:
        st.markdown(f"**Outputs of run {run} ({pick})**")
        st.dataframe(pd.DataFrame({
            "Output": g["keys"], "Value": [row_out[k] for k in g["keys"]], "Unit": [units.get(k, "") for k in g["keys"]]}),
            hide_index=True, width="stretch", column_config={"Value": st.column_config.NumberColumn(format="%.5g")})
    st.markdown("**Top 10 runs**")
    show(gra_rank_fig(g["ranking"], series_color(pick, order)))

    st.markdown("##### What moves the grade")
    show(contrib_fig({n: x["table"] for n, x in gra.items()}, order, "Contribution to variation in the GRA grade"))
    st.markdown("##### Best settings for the combined grade")
    for n, x in gra.items():
        opt = x["optimum"]
        st.markdown(f"**{n}**  ·  predicted grade {opt['predicted']:.3f}"
                    + (f", which is run {opt['tested_run']}" if opt["tested_run"] is not None else ", not among the tested runs"))
        st.dataframe(opt["setting"], hide_index=True, width="stretch", column_config={
            "Mean at that level": st.column_config.NumberColumn(format="%.4g"),
            "Effect vs overall mean": st.column_config.NumberColumn(format="%+.3g")})
    with st.expander("Full ranking of all runs"):
        st.dataframe(g["ranking"], hide_index=True, width="stretch", height=360)


def render_export(analysis: dict, findings: list[dict]) -> None:
    st.markdown("##### Download")
    wide = design.copy()
    wide.columns = [RUN] + [f.label for f in factors]
    for ds, d in datasets.items():
        for c in d.columns:
            if c != RUN:
                wide[f"{ds} | {dc.ulabel(c, units)}"] = d[c].to_numpy()
    e1, e2 = st.columns(2)
    e1.download_button("Everything as one Excel workbook", dc.export_workbook(design, factors, datasets, analysis, findings, units),
                       file_name="doe_analysis.xlsx", icon=":material/table:", width="stretch", mime=XLSX)
    e2.download_button("Design and all results (CSV)", wide.to_csv(index=False), file_name="doe_design_and_results.csv",
                       mime="text/csv", icon=":material/download:", width="stretch")
    st.markdown("##### Design and results, all runs")
    st.dataframe(wide, hide_index=True, width="stretch", height=420)


with tab_res:
    if design is None:
        st.warning("Define the design in step 1 first.", icon=":material/arrow_back:")
    elif not datasets:
        st.info("Add results in step 2. Use the demo data there to see the full layout.", icon=":material/arrow_back:")
    elif not selected:
        st.info("Pick at least one output in the sidebar.", icon=":material/arrow_back:")
    else:
        analysis = dc.analyze(design, factors, datasets, selected, goals, reference, detailed, gra_cfg, zeta)
        findings = dc.build_findings(design, factors, analysis, units, goals, reference, design_meta.get("array"))
        st.caption("Change the outputs, goals and reference in the sidebar. Everything below updates.")
        t_sum, t_cmp, t_drv, t_opt, t_gra, t_exp = st.tabs(
            ["Summary", "Compare sources", "Factor drivers", "Best settings", "Best run (GRA)", "Tables and export"])
        with t_sum:
            render_summary(analysis, findings)
        with t_cmp:
            render_compare(analysis)
        with t_drv:
            render_drivers(analysis)
        with t_opt:
            render_settings(analysis)
        with t_gra:
            render_gra(analysis)
        with t_exp:
            render_export(analysis, findings)

st.divider()
st.caption("This app holds no physics model: every number comes from the results you enter. Contribution percentages describe "
           "main effects over the levels you tested, and predicted best settings assume effects add up, so confirm them with a run.")
