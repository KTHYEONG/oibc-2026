import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.oibc.compose.constraints import CompositionError
from src.oibc.compose.store import assignment_sha256, load_composition, write_composition
from src.oibc.config import ArtifactPaths, CompetitionSpec, ComposeConfig


def _sites() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": [f"S{i:03d}" for i in range(1, 7)],
            "capacity_kw": [500.0, 500.0, 500.0, 500.0, 500.0, 500.0],
        }
    )


def _assignment() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": [f"S{i:03d}" for i in range(1, 7)],
            "group": ["G1", "G1", "G1", "G2", "G2", "G2"],
        }
    )


def _spec() -> CompetitionSpec:
    return CompetitionSpec(n_sites=6, min_group_capacity_kw=1000.0)


def _paths(tmp_path: Path) -> ArtifactPaths:
    return ArtifactPaths(compositions=tmp_path / "compositions")


def test_assignment_sha256_order_invariant() -> None:
    base = _assignment()
    shuffled = base.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert assignment_sha256(base) == assignment_sha256(shuffled)


def test_assignment_sha256_changes_on_group_flip() -> None:
    base = _assignment()
    flipped = base.copy()
    flipped.loc[0, "group"] = "G2"
    assert assignment_sha256(base) != assignment_sha256(flipped)


def test_write_then_load_roundtrip(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    csv_path = write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    assert csv_path == paths.compositions / "geo64.csv"
    loaded, digest = load_composition("geo64", _sites(), paths=paths, spec=_spec())
    pd.testing.assert_frame_equal(loaded, _assignment().astype("string"))
    assert digest == assignment_sha256(_assignment())
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == digest


def test_write_composition_refuses_overwrite(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    csv_path = paths.compositions / "geo64.csv"
    before = csv_path.read_bytes()
    with pytest.raises(CompositionError):
        write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    assert csv_path.read_bytes() == before


def test_write_composition_rejects_reserved_and_unsafe_id(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    for bad in ("current_group", "../x", "", ".", "..", "a/b"):
        with pytest.raises(CompositionError):
            write_composition(_assignment(), bad, ComposeConfig(), _sites(), paths=paths, spec=_spec())
    assert not paths.compositions.exists() or list(paths.compositions.iterdir()) == []


def test_write_composition_rejects_invalid_assignment(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    bad = _assignment().copy()
    bad.loc[0, "group"] = "G2"
    bad.loc[1, "group"] = "G2"
    bad.loc[2, "group"] = "G2"
    with pytest.raises(CompositionError):
        write_composition(bad, "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    assert not (paths.compositions / "geo64.csv").exists()
    assert not (paths.compositions / "geo64.json").exists()


def test_load_composition_detects_tampering(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    csv_path = paths.compositions / "geo64.csv"
    text = csv_path.read_text(encoding="utf-8").replace("S001,G1", "S001,G2")
    csv_path.write_text(text, encoding="utf-8")
    with pytest.raises(CompositionError):
        load_composition("geo64", _sites(), paths=paths, spec=_spec())


def test_load_composition_requires_both_files(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    (paths.compositions / "geo64.json").unlink()
    with pytest.raises(CompositionError):
        load_composition("geo64", _sites(), paths=paths, spec=_spec())


def test_load_composition_rejects_corrupt_record(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    (paths.compositions / "geo64.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CompositionError):
        load_composition("geo64", _sites(), paths=paths, spec=_spec())


def test_load_composition_rejects_non_utf8_payload(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    csv_path = paths.compositions / "geo64.csv"
    payload = csv_path.read_bytes() + b"\xff\xfe"
    csv_path.write_bytes(payload)
    json_path = paths.compositions / "geo64.json"
    record = json.loads(json_path.read_text(encoding="utf-8"))
    record["assignment_sha256"] = hashlib.sha256(payload).hexdigest()
    json_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CompositionError):
        load_composition("geo64", _sites(), paths=paths, spec=_spec())


def test_load_composition_revalidates_against_sites(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    grown = pd.concat(
        [pd.DataFrame({"site_id": ["S000"], "capacity_kw": [10.0]}), _sites()], ignore_index=True
    )
    grown_spec = CompetitionSpec(n_sites=7, min_group_capacity_kw=1000.0)
    with pytest.raises(CompositionError):
        load_composition("geo64", grown, paths=paths, spec=grown_spec)


def test_assignment_sha256_rejects_bad_columns() -> None:
    bad = _assignment().rename(columns={"group": "grp"})
    with pytest.raises(CompositionError):
        assignment_sha256(bad)


def test_assignment_sha256_rejects_null_value() -> None:
    bad = _assignment()
    bad.loc[0, "group"] = None
    with pytest.raises(CompositionError):
        assignment_sha256(bad)


def test_write_composition_without_git_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    paths = _paths(tmp_path)
    write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    record = json.loads((paths.compositions / "geo64.json").read_text(encoding="utf-8"))
    assert record["git_commit"] is None


def test_write_composition_without_git_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def _boom(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("no git")

    monkeypatch.setattr(subprocess, "run", _boom)
    paths = _paths(tmp_path)
    write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    record = json.loads((paths.compositions / "geo64.json").read_text(encoding="utf-8"))
    assert record["git_commit"] is None


def test_write_composition_cleans_temp_files_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json as json_module

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(json_module, "dump", _boom)
    paths = _paths(tmp_path)
    with pytest.raises(RuntimeError, match="disk full"):
        write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    assert list(paths.compositions.iterdir()) == []


def test_write_composition_rolls_back_second_publish_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    replace = Path.replace

    def fail_json(source: Path, target: Path) -> Path:
        if target.name == "geo64.json":
            raise OSError("publish failed")
        return replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_json)
        with pytest.raises(OSError, match="publish failed"):
            write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    assert list(paths.compositions.iterdir()) == []
    write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    load_composition("geo64", _sites(), paths=paths, spec=_spec())


def test_concurrent_writers_never_overwrite(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    paths = _paths(tmp_path)

    def write(index: int) -> bool:
        assignment = _assignment()
        if index:
            assignment["group"] = assignment["group"].map({"G1": "G2", "G2": "G1"})
        try:
            write_composition(assignment, "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
        except CompositionError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        succeeded = list(executor.map(write, range(2)))
    assert sum(succeeded) == 1
    loaded, digest = load_composition("geo64", _sites(), paths=paths, spec=_spec())
    expected = _assignment()
    if succeeded[1]:
        expected["group"] = expected["group"].map({"G1": "G2", "G2": "G1"})
    pd.testing.assert_frame_equal(loaded, expected.astype("string"))
    assert digest == assignment_sha256(expected)


def test_write_refuses_orphan_json_and_dangling_symlink(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.compositions.mkdir()
    orphan = paths.compositions / "geo64.json"
    orphan.write_text("orphan", encoding="utf-8")
    with pytest.raises(CompositionError, match="already exists"):
        write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    assert orphan.read_text(encoding="utf-8") == "orphan"
    link = paths.compositions / "other.csv"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(CompositionError, match="already exists"):
        write_composition(_assignment(), "other", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    assert link.is_symlink()


def test_load_rejects_reserved_id(tmp_path: Path) -> None:
    with pytest.raises(CompositionError, match="reserved"):
        load_composition("current_group", _sites(), paths=_paths(tmp_path), spec=_spec())


def test_load_rejects_mismatched_record_id(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    path = paths.compositions / "geo64.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["comp_id"] = "other"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CompositionError, match="record id mismatch"):
        load_composition("geo64", _sites(), paths=paths, spec=_spec())


@pytest.mark.parametrize("payload", [b"", b'site_id,group\n"unterminated', b"group,site_id\nG1,S001\n"])
def test_load_rejects_malformed_csv_even_with_updated_digest(tmp_path: Path, payload: bytes) -> None:
    paths = _paths(tmp_path)
    write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    (paths.compositions / "geo64.csv").write_bytes(payload)
    path = paths.compositions / "geo64.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["assignment_sha256"] = hashlib.sha256(payload).hexdigest()
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CompositionError):
        load_composition("geo64", _sites(), paths=paths, spec=_spec())


def test_load_checks_canonical_digest(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    write_composition(_assignment(), "geo64", ComposeConfig(), _sites(), paths=paths, spec=_spec())
    payload = _assignment().iloc[::-1].to_csv(index=False).encode("utf-8")
    (paths.compositions / "geo64.csv").write_bytes(payload)
    path = paths.compositions / "geo64.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["assignment_sha256"] = hashlib.sha256(payload).hexdigest()
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CompositionError, match="noncanonical"):
        load_composition("geo64", _sites(), paths=paths, spec=_spec())


def test_store_preserves_csv_na_like_site_ids(tmp_path: Path) -> None:
    sites = _sites()
    sites.loc[0, "site_id"] = "NA"
    assignment = _assignment()
    assignment.loc[0, "site_id"] = "NA"
    paths = _paths(tmp_path)
    write_composition(assignment, "geo64", ComposeConfig(), sites, paths=paths, spec=_spec())
    loaded, digest = load_composition("geo64", sites, paths=paths, spec=_spec())
    assert "NA" in loaded.site_id.tolist()
    assert digest == assignment_sha256(loaded)
