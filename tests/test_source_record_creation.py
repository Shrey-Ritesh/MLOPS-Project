from src.data.inventory import build_source_record, write_jsonl

def test_source_record_creation():
    jee = build_source_record(
        r"data/raw/pyq/JEE Main 2025 (28 Jan Shift 2) Previous Year Paper with Answer Keys - MathonGo.pdf",
        source_type="jee_main",
    )

    ncert = build_source_record(
        r"data/raw/ncert/lech205.pdf",
        source_type="ncert",
    )

    records = [jee, ncert]

    write_jsonl(records, "data/processed/source_manifest.jsonl")

    assert len(records) == 2