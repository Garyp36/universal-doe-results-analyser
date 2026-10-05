# DOE Results Analyzer

A Streamlit app for design-of-experiments studies where you produce the results yourself, in CFD or on a test rig. It has no physics model. You define the factors and levels, run the cases, enter the outputs, and the app does the statistics: comparison between sources, factor contribution (ANOVA), best settings, and grey relational analysis (GRA) for several outputs at once.

It works for any study, including a new crossflow-radiator arrangement. Use one output or many, and one source (CFD only, test only) or up to three.

## How it works

| Step | What happens |
|---|---|
| 1. Design | List factors, units and levels. The app picks the smallest Taguchi orthogonal array that carries them (or a full factorial) and builds the run list. Or load a design table you already have. |
| 2. Results data | Name your outputs, download the run sheet, run the cases, then type the values into the table, upload the filled sheet (CSV or Excel), or paste from Excel. Up to three datasets, for example CFD and Experiment. Runs may stay empty and be filled later. |
| Sidebar | Choose which outputs to analyse, the reference dataset, the goal for each output (higher, lower or closest to a target) and the GRA weights. |
| 3. Analysis | Summary with findings, source comparison, factor drivers, best settings, best run by GRA, and an Excel export. |

Colored text marks what matters: blue for key numbers and terms, orange for offsets and cautions, green for good agreement or the best run, violet for the leading factor.

## Files

```
app.py                  the Streamlit interface
doe_core.py             arrays, parsing, ANOVA, comparison, GRA, findings, export (no Streamlit code)
requirements.txt        Python packages
.streamlit/config.toml  theme (light and dark)
tests/                  pytest checks for the core and the app
.github/workflows/      runs the tests on every push
```

## Run it on your computer

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run the tests with `pip install pytest` then `pytest -q`.

## Put it on GitHub and publish it

1. Create a new repository on github.com (for example `doe-results-analyzer`). Public repositories deploy for free.
2. Upload every file and folder from this project and keep the structure above. On the repository page choose Add file, then Upload files, and drag the whole folder in. Check that `.streamlit/config.toml` and `.github/workflows/tests.yml` are included (folders that start with a dot are hidden on some systems).
3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
4. Choose Create app, pick your repository, set the branch to `main` and the main file to `app.py`, then deploy.
5. The first build takes a few minutes. Every later push to GitHub updates the app.

## Designs the app can build

| Array | Runs | Columns |
|---|---|---|
| L4, L8, L16, L32 | 4, 8, 16, 32 | two-level, 3, 7, 15 and 31 factors |
| L12 | 12 | two-level, up to 11 factors |
| L9, L27 | 9, 27 | three-level, up to 4 and 13 factors |
| L16 (4⁵) | 16 | four-level, up to 5 factors |
| L18 | 18 | one two-level and up to seven three-level factors |
| L25 | 25 | five-level, up to 6 factors |
| L49 | 49 | seven-level, up to 8 factors |
| L50 | 50 | one two-level and up to eleven five-level factors |
| Full factorial | up to 300 | every combination |

Mixed levels work: a factor with fewer levels than its array column gets dummy levels (the first levels repeat), and the app flags the resulting imbalance. The 4-level arrays use GF(4) arithmetic, and every array in the catalog is checked for balance and orthogonality by the tests.

If you already have a design (any array, a Latin hypercube, a hand-made table), choose "I already have a design table". Factors with up to 8 distinct values are compared level by level. Factors with more are fitted as a straight-line trend, which you can switch per factor.

## File format for results

One row per run. Factor columns are optional and ignored (the design table is the source of truth), so you can fill in the downloaded run sheet and upload it unchanged.

| Column | Meaning |
|---|---|
| `Run` | Run number from the design. Without it, rows are read in design order and the row count must match. |
| `Heat transfer rate (W)` and any other output | One numeric column per output, unit in brackets. The same output in two datasets is matched by name, ignoring capitals, spaces and punctuation. |
| `Suggested order`, `Notes` | Ignored. |

Empty cells are allowed. Repeated run numbers (replicates) are averaged. Text in a numeric column is rejected with the column and row.

## How to read the results

- NRMSE is the root-mean-square difference between a dataset and the reference as a percentage of the reference range. Under 5% reads as good, 5 to 15% as moderate, above 15% as poor. These bands are a rule of thumb, not a standard.
- Mean error is the dataset minus the reference. Positive means the dataset reads higher.
- Rank correlation (Spearman ρ) says whether two sources order the runs the same way, which matters more for choosing settings than the absolute offset does.
- Contribution (%) is the share of an output's variation that one factor explains. It uses a drop-one sum of squares, so it also works for unbalanced designs, partly filled results and straight-line factors. For an orthogonal design the factor shares and the unexplained share add to 100%.
- F and p-value test each factor against the leftover variation. When a design leaves fewer than 2 error degrees of freedom (saturated arrays such as L25 with six factors), the weakest factors are pooled into error, as in Taguchi practice, and the table marks them. With CFD data these are screening aids, because a deterministic solver has no random noise.
- Predicted best adds up the best level of each factor that explains at least 5% of the variation. It assumes effects add up, so confirm it with an extra run.
- GRA grade merges several outputs into one score per run against your goals and weights. The best run is the best of the tested settings, not a continuous optimum. Outputs linked by physics (heat rate and outlet temperature, for example) count the same effect twice, so give them smaller weights.

## Checked by tests

The tests in `tests/` cover every orthogonal array, the F distribution against SciPy, the ANOVA against the textbook balanced formula, unbalanced and saturated designs, straight-line factors, file parsing (aliases, partial results, replicates, bad input), comparison metrics, GRA, the findings, the Excel export, and the app itself.
