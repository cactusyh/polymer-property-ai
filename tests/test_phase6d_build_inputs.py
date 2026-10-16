import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase6d_build_inputs as build_inputs
import phase6d_ready_queue as ready_queue


def test_input_template_preserves_identity_and_temperature():
    text = build_inputs.input_text("poly_test", 400)
    assert "# polymer_id=poly_test" in text
    assert "temp 400.0 400.0" in text
    assert "production_400K" in text
    assert "-in input_400K.in" in text


def test_ready_queue_rejects_missing_input(tmp_path):
    errors = ready_queue.validate_input(tmp_path / "missing.in", "poly_test", 300)
    assert errors == ["missing_input"]


def test_ready_queue_rejects_identity_and_temperature(tmp_path):
    work = tmp_path / "poly_test" / "300K"
    work.mkdir(parents=True)
    (work / "structure.data").write_text("1 atoms\n", encoding="utf-8")
    path = work / "input_300K.in"
    path.write_text(build_inputs.input_text("poly_other", 400), encoding="utf-8")
    errors = ready_queue.validate_input(path, "poly_test", 300)
    assert "identity_mismatch" in errors
    assert "temperature_mismatch" in errors


def test_structure_hash_is_deterministic():
    smiles = "*CC*"
    assert build_inputs.text_sha(smiles) == build_inputs.text_sha(smiles)
