import json

import pymupdf
import pytest

from src.data.ids import make_block_id, make_chapter_id
from src.data.ncert_clean import clean_block_text, private_use_ratio
from src.data.ncert_extract import extract_chapter
from src.data.ncert_pipeline import validate_dataset
from src.models import NCERTChapterRecord


@pytest.fixture()
def chapter_pdf(tmp_path):
    doc = pymupdf.open()
    for i in range(1, 4):
        page = doc.new_page(width=600, height=800)
        page.insert_text((250, 40), f"{i + 100}", fontsize=8, fontname="helv")  # page number
        page.insert_text((250, 780), "Reprint 2026-27", fontsize=8, fontname="helv")  # footer
        if i == 1:
            page.insert_text((200, 110), "CHAPTER EIGHT", fontsize=16, fontname="hebo")
            page.insert_text((100, 180), "MECHANICAL PROPERTIES", fontsize=20, fontname="hebo")
        page.insert_text((72, 260), f"{i}.1 Stress and strain", fontsize=11, fontname="hebo")
        page.insert_text((72, 300), "A solid has a definite shape and size and well-", fontsize=10, fontname="helv")
        page.insert_text((72, 314), "known elastic behaviour.", fontsize=10, fontname="helv")
        page.insert_text((72, 400), "F/A = Y x dL/L (8.1)", fontsize=10, fontname="helv")
        page.insert_text((72, 500), "Fig. 8.1 A cylindrical body", fontsize=9, fontname="helv")
    path = tmp_path / "ch.pdf"
    doc.save(path)
    return str(path)


def _extract(path):
    return extract_chapter(path, source_id="src-" + "a" * 64, subject="Physics", class_level=11, language="en")


def test_chapter_record_validates_against_contract(chapter_pdf):
    result = _extract(chapter_pdf)
    record = NCERTChapterRecord.model_validate(result.record)
    assert record.chapter_number == "8"
    assert record.chapter_title == "MECHANICAL PROPERTIES"
    assert (record.page_start, record.page_end) == (1, 3)
    assert record.extraction_status.value == "success"


def test_blocks_typed_ordered_and_boilerplate_removed(chapter_pdf):
    blocks = _extract(chapter_pdf).record["content_blocks"]
    assert [b["order_index"] for b in blocks] == list(range(len(blocks)))
    types = {b["block_type"] for b in blocks}
    assert {"heading", "paragraph", "equation", "caption"} <= types
    joined = " ".join(b["text_cleaned"] or "" for b in blocks)
    assert "Reprint" not in joined
    eq = next(b for b in blocks if b["block_type"] == "equation")
    assert eq["original_label"] == "(8.1)"
    assert "well-known" in joined  # hyphen kept, line break removed


def test_text_raw_keeps_original_and_ids_are_deterministic(chapter_pdf):
    a, b = _extract(chapter_pdf), _extract(chapter_pdf)
    assert "Reprint 2026-27" in a.record["text_raw"]
    assert a.record == b.record


def test_failed_extraction_when_no_text(tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    path = tmp_path / "blank.pdf"
    doc.save(path)
    result = _extract(str(path))
    assert result.record["extraction_status"] == "failed"
    assert result.record["content_blocks"] == []
    NCERTChapterRecord.model_validate(result.record)


def test_clean_block_text_rules():
    assert clean_block_text("a  b\nc", "paragraph") == "a b c"
    assert clean_block_text("x = 1\ny = 2", "equation") == "x = 1\ny = 2"
    assert clean_block_text("−5 × 10−3", "paragraph") == "−5 × 10−3"
    assert clean_block_text(None, "figure") is None
    assert private_use_ratio("ab") == pytest.approx(1 / 3)


def test_ids_stable_and_path_independent():
    cid = make_chapter_id("src-abc", "8")
    assert cid == make_chapter_id("src-abc", " 8 ")
    assert make_block_id(cid, 3) == f"{cid}-b00003"
    with pytest.raises(ValueError):
        make_chapter_id("", "8")


def test_dataset_validation_flags_problems(chapter_pdf):
    rec = _extract(chapter_pdf).record
    assert validate_dataset([rec], {rec["source_id"]: "ncert"}) == []
    errors = validate_dataset([rec, rec], {})
    assert any("duplicate" in e for e in errors)
    assert any("not an NCERT source" in e for e in errors)


def _blocks(path):
    return _extract(path).record["content_blocks"]


def test_stacked_fraction_is_rebuilt_into_one_line(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((100, 306), "Y =", fontsize=10, fontname="helv")
    page.insert_text((130, 300), "F", fontsize=10, fontname="helv")
    page.insert_text((130, 313), "A", fontsize=10, fontname="helv")
    page.insert_text((300, 306), "(8.1)", fontsize=10, fontname="helv")
    page.insert_text((72, 500), "Plain text paragraph here.", fontsize=10, fontname="helv")
    path = tmp_path / "frac.pdf"
    doc.save(path)
    eq = next(b for b in _blocks(str(path)) if b["block_type"] == "equation")
    assert eq["text_cleaned"] == "Y = F/A (8.1)"
    assert eq["original_label"] == "(8.1)"


def test_table_rebuilt_from_filled_background(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((72, 100), "Table 8.2 Shear moduli of materials", fontsize=10, fontname="hebo")
    page.draw_rect(pymupdf.Rect(70, 120, 300, 200), color=None, fill=(0.8, 0.9, 1.0))
    for y, (a, b) in zip((135, 150, 165, 180), [("Material", "G"), ("Brass", "36"), ("Iron", "70"), ("Lead", "5.6")]):
        page.insert_text((80, y), a, fontsize=10, fontname="helv")
        page.insert_text((220, y), b, fontsize=10, fontname="helv")
    path = tmp_path / "table.pdf"
    doc.save(path)
    result = _extract(str(path))
    table = next(b for b in result.record["content_blocks"] if b["block_type"] == "table")
    assert table["text_cleaned"].split("\n") == ["Material | G", "Brass | 36", "Iron | 70", "Lead | 5.6"]
    assert result.stats["tables_reconstructed"] == 1
    caption = next(b for b in result.record["content_blocks"] if b["block_type"] == "caption")
    assert caption["original_label"] == "Table 8.2"


def test_in_text_table_mention_is_not_a_caption(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((72, 200), "Table 8.4 shows the various types of stress.", fontsize=10, fontname="helv")
    path = tmp_path / "mention.pdf"
    doc.save(path)
    assert all(b["block_type"] != "caption" for b in _blocks(str(path)))


def test_two_column_page_reads_left_column_first(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    for i in range(5):  # right column text starts higher than the lower left lines
        page.insert_text((330, 150 + i * 60), f"RIGHT{i} " + "word " * 10, fontsize=10, fontname="helv")
    for i in range(5):
        page.insert_text((60, 160 + i * 60), f"LEFT{i} " + "word " * 10, fontsize=10, fontname="helv")
    path = tmp_path / "cols.pdf"
    doc.save(path)
    order = [b["text_cleaned"].split()[0] for b in _blocks(str(path))]
    assert order == [f"LEFT{i}" for i in range(5)] + [f"RIGHT{i}" for i in range(5)]


def test_private_use_codes_are_decoded():
    from src.data.ncert_clean import decode_private_use

    assert decode_private_use("") == "CHAP"
    assert decode_private_use("earths") == "earth’s"
    assert decode_private_use("", symbol_font=True) == "αβ"
    assert decode_private_use("plain text") == "plain text"
