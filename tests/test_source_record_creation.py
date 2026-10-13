from pathlib import Path
from src.data.inventory import build_source_record, write_jsonl


def test_source_record_creation(tmp_path):
    pdf_files = sorted(Path("data/raw").glob("*.pdf"))
    assert len(pdf_files) > 0, "No raw PDF files found in data/raw"

    jee = build_source_record(
        pdf_files[0],
        source_type="jee_main",
    )

    dummy_doc = tmp_path / "sample_doc.pdf"
    dummy_doc.write_bytes(b"%PDF-1.4 sample file content for testing")
    ncert = build_source_record(
        dummy_doc,
        source_type="ncert",
    )

    records = [jee, ncert]
    test_out = tmp_path / "test_source_manifest.jsonl"
    write_jsonl(records, test_out)

    assert len(records) == 2
    assert test_out.exists()