import json
from pathlib import Path
from transformers import AutoTokenizer

DATA_PATH = Path("data/processed/ncert_learning_units.jsonl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256  # Includes special tokens

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

units = []

with DATA_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            units.append(json.loads(line))

results = []

for unit in units:
    # Match the text template used by your index builder.
    text = f"""Subject: {unit.get('subject', '')}
Class: {unit.get('class_level', '')}
Chapter: {unit.get('chapter_title', '')}
Topic: {unit.get('unit_title', '')}

{unit.get('text', '')}"""

    # Count tokens WITHOUT truncating.
    token_count = len(
        tokenizer.encode(text, add_special_tokens=True,
                         truncation=False)
    )

    results.append({
        "unit_id": unit["unit_id"],
        "unit_title": unit.get("unit_title", ""),
        "tokens": token_count,
        "over_limit": token_count > MAX_TOKENS,
    })

results.sort(key=lambda x: x["tokens"], reverse=True)

over_limit = [r for r in results if r["over_limit"]]

print(f"Total units: {len(results)}")
print(f"Units over {MAX_TOKENS} tokens: {len(over_limit)}")
print(f"Longest unit: {results[0]['tokens']} tokens")

print("\\nTop 15 longest units:")
for r in results[:15]:
    print(f"{r['tokens']:>5} tokens | {r['unit_id']} | {r['unit_title']}")

print("\\nUnits exceeding the limit:")
for r in over_limit:
    print(f"{r['tokens']:>5} tokens | {r['unit_id']} | {r['unit_title']}")

Path("reports/retrieval").mkdir(parents=True, exist_ok=True)
with open("reports/retrieval/unit_length_report.json", "w",
          encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\\nFull report saved to reports/retrieval/unit_length_report.json")