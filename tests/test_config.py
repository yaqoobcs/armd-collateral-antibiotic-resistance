import os

import pytest

from armd.config import Config, check_required, missing_tables, resolve_data_dir


def test_both_spellings_of_the_data_directory_resolve(tmp_path):
    """Both spellings of the data directory, ARMD_dataset and AMRD_dataset, resolve."""
    (tmp_path / "AMRD_dataset").mkdir()
    assert resolve_data_dir(root=str(tmp_path)).endswith("AMRD_dataset")

    other = tmp_path / "elsewhere"
    (other / "ARMD_dataset").mkdir(parents=True)
    assert resolve_data_dir(root=str(other)).endswith("ARMD_dataset")


def test_a_missing_directory_is_an_error_not_a_none(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_data_dir(root=str(tmp_path / "nothing_here"))
    with pytest.raises(FileNotFoundError):
        resolve_data_dir(explicit=str(tmp_path / "nope"))


def test_required_table_check_names_the_file(tmp_path):
    d = tmp_path / "AMRD_dataset"
    d.mkdir()
    with pytest.raises(FileNotFoundError) as exc:
        check_required(str(d))
    assert "microbiology_cultures_cohort.csv" in str(exc.value)
    assert "--smoke" in str(exc.value)


def test_optional_tables_are_listed_not_fatal(tmp_path):
    d = tmp_path / "AMRD_dataset"
    d.mkdir()
    (d / "microbiology_cultures_cohort.csv").write_text("x\n")
    check_required(str(d))
    assert "microbiology_cultures_labs.csv" in missing_tables(
        str(d), ["microbiology_cultures_labs.csv"]
    )


def test_config_round_trips_through_json(tmp_path):
    cfg = Config(target_class="Cephalosporin", n_folds=7)
    path = str(tmp_path / "c.json")
    cfg.to_json(path)
    assert Config.from_json(path).target_class == "Cephalosporin"
    assert Config.from_json(path).n_folds == 7


def test_cli_flags_override_the_config_file(tmp_path):
    from armd.run_analysis import build_parser

    args = build_parser().parse_args(["--target-class", "Carbapenem", "--folds", "3"])
    assert args.target_class == "Carbapenem"
    assert args.folds == 3


def test_cli_reports_a_missing_cohort_table_instead_of_crashing(tmp_path, capsys):
    from armd.run_analysis import main

    d = tmp_path / "AMRD_dataset"
    d.mkdir()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        code = main(["--quiet"])
    finally:
        os.chdir(cwd)
    assert code == 2
    assert "microbiology_cultures_cohort.csv" in capsys.readouterr().err
