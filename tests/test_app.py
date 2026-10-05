"""Smoke tests that run the whole Streamlit script headlessly. Run with:  pytest -q"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")

OWN_DESIGN = """Run\tVelocity (m/s)\tFin\tTube pitch (mm)
1\t2\tA\t25
2\t2\tB\t35
3\t2\tC\t45
4\t4\tA\t35
5\t4\tB\t45
6\t4\tC\t25
7\t6\tA\t45
8\t6\tB\t25
9\t6\tC\t35
"""


@pytest.fixture()
def app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def outputs_box(at):
    return [m for m in at.multiselect if m.label == "Outputs to analyse"][0]


def test_loads_with_the_example_design(app):
    assert not app.exception and not app.error
    assert any("Balanced and orthogonal" in s.value for s in app.success)
    assert [m.value for m in app.metric][:2] == ["25", "4"]


def test_demo_data_unlocks_analysis(app):
    app.radio(key="ds_src_0").set_value("Demo data").run()
    assert not app.exception and not app.error
    assert "Air pressure drop (Pa)" in outputs_box(app).options
    assert any("Best run by combined GRA grade" in s.value for s in app.success)


def test_two_datasets_compare_and_agree_on_best_run(app):
    app.radio(key="ds_src_0").set_value("Demo data").run()
    app.segmented_control(key="n_ds").set_value(2).run()
    app.radio(key="ds_src_1").set_value("Demo data").run()
    assert not app.exception and not app.error
    assert any("All datasets agree" in s.value for s in app.success)


def test_single_output_skips_gra(app):
    app.radio(key="ds_src_0").set_value("Demo data").run()
    box = outputs_box(app)
    box.set_value([box.options[0]]).run()
    assert not app.exception and not app.error
    assert not any("GRA grade" in s.value for s in app.success)


def test_duplicate_dataset_name_is_flagged(app):
    app.radio(key="ds_src_0").set_value("Demo data").run()
    app.segmented_control(key="n_ds").set_value(2).run()
    app.text_input(key="ds_name_1").set_value("CFD").run()
    app.radio(key="ds_src_1").set_value("Demo data").run()
    assert any("already used" in e.value for e in app.error)


def test_own_design_table_pasted_from_excel(app):
    app.radio(key="design_mode").set_value("I already have a design table").run()
    app.radio(key="design_src").set_value("Paste from Excel").run()
    app.text_area(key="design_text").set_value(OWN_DESIGN).run()
    assert not app.exception and not app.error
    assert [m.value for m in app.metric][:2] == ["9", "3"]
    app.radio(key="ds_src_0").set_value("Demo data").run()
    assert not app.exception and not app.error
    assert any("run" in s.value.lower() for s in app.success)


def test_pasted_results_with_a_mistake_are_reported(app):
    app.radio(key="ds_src_0").set_value("Paste from Excel").run()
    app.text_area(key="ds_text_0").set_value("Run\tHeat transfer rate (W)\n1\t100\n2\tabc\n").run()
    assert not app.exception
    assert any("non-numeric" in e.value or "rows are read in order" in e.value for e in app.error)
