from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


CATALOG_FILE = Path(__file__).with_name("reasoning_templates.json")
LETTER_PATTERN = re.compile(r"\{([ABCD])\}")


LANGUAGES: dict[str, dict[str, Any]] = {
    "telugu": {
        "key": "telugu",
        "question_label": "ప్రశ్న",
        "reasoning_label": "తర్కం",
        "answer_label": "జవాబు",
        "attributes": ["ఎత్తు", "వయస్సు", "ధర", "బరువు", "దూరం"],
        "names": {
            "train": ["రవి", "సీత", "కిరణ్", "లక్ష్మి", "మోహన్", "కావ్య", "అనిల్", "దీపిక", "వికాస్", "ప్రియా", "సురేష్", "నందిని", "అర్జున్", "మేఘన", "హరి", "వాణి"],
            "validation": ["అమర్", "స్వాతి", "రాజ్", "స్నేహ", "కార్తిక్", "పద్మ", "భరత్", "మాలతి"],
            "test": ["మధు", "గాయత్రి", "రోహిత్", "జ్యోతి", "నవీన్", "సరిత", "చరణ్", "లీలా"]
        }
    },
    "nepali": {
        "key": "nepali",
        "question_label": "प्रश्न",
        "reasoning_label": "तर्क",
        "answer_label": "उत्तर",
        "attributes": ["उचाइ", "उमेर", "मूल्य", "तौल", "दूरी"],
        "names": {
            "train": ["रवि", "सीता", "किरण", "लक्ष्मी", "मोहन", "काव्या", "अनिल", "दीपिका", "विकास", "प्रिया", "सुरेश", "नन्दिनी", "अर्जुन", "मेघा", "हरि", "वाणी"],
            "validation": ["अमर", "स्वाती", "राज", "स्नेहा", "कार्तिक", "पद्मा", "भरत", "मालती"],
            "test": ["मधु", "गायत्री", "रोहित", "ज्योति", "नवीन", "सरिता", "चरण", "लीला"]
        }
    }
}


def required_letters(template: dict[str, Any], language: str) -> list[str]:
    """Return the entity slots used by a template in a stable order."""
    rendered = template[language]
    text = " ".join(rendered["facts"] + [rendered["question"], rendered["reasoning"], rendered["answer"]])
    return sorted(set(LETTER_PATTERN.findall(text)))


def render_record(
    template: dict[str, Any],
    language_name: str,
    split: str,
    index: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Instantiate one template and return a model-ready, auditable record."""
    language = LANGUAGES[language_name]
    slots = required_letters(template, language["key"])
    values = rng.sample(language["names"][split], k=len(slots))
    substitutions = dict(zip(slots, values))
    substitutions["attribute"] = rng.choice(language["attributes"])

    def fill(text: str) -> str:
        return text.format(**substitutions)

    localized = template[language["key"]]
    facts = [fill(fact) for fact in localized["facts"]]
    question = fill(localized["question"])
    reasoning = fill(localized["reasoning"])
    answer = fill(localized["answer"])
    facts_text = " ".join(facts)
    prompt = f"{language['question_label']}: {facts_text} {question}\n{language['reasoning_label']}:"
    target = f" {reasoning}\n{language['answer_label']}: {answer}"

    return {
        "id": f"{language_name[:2]}_{split[:3]}_{index:06d}",
        "language": language_name,
        "split": split,
        "template_id": template["id"],
        "family": template["family"],
        "logic": template["logic"],
        "entities": {slot: substitutions[slot] for slot in slots},
        "attribute": substitutions["attribute"],
        "facts": facts,
        "question": question,
        "reasoning": reasoning,
        "answer": answer,
        "prompt": prompt,
        "target": target,
        "text": prompt + target
    }


def generate_split(
    templates: list[dict[str, Any]],
    language_name: str,
    split: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Generate a template-balanced split with no duplicate model text."""
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    attempts = 0
    while len(records) < count:
        template = templates[len(records) % len(templates)]
        record = render_record(template, language_name, split, len(records), rng)
        attempts += 1
        if record["text"] in seen_text:
            if attempts > count * 50:
                raise RuntimeError(f"Could not produce {count} unique {language_name} {split} records.")
            continue
        seen_text.add(record["text"])
        records.append(record)
    rng.shuffle(records)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write UTF-8 JSONL without ASCII escaping so each script remains readable."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("reasoning_dataset"))
    parser.add_argument("--train-count", type=int, default=20_000)
    parser.add_argument("--validation-count", type=int, default=2_500)
    parser.add_argument("--test-count", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    catalog = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    templates = catalog["templates"]
    split_counts = {"train": args.train_count, "validation": args.validation_count, "test": args.test_count}
    manifest: dict[str, Any] = {
        "catalog_version": catalog["catalog_version"],
        "base_seed": args.seed,
        "template_count": len(templates),
        "splits": split_counts,
        "languages": {}
    }

    for language_offset, language_name in enumerate(LANGUAGES):
        language_dir = args.output_dir / language_name
        language_dir.mkdir(parents=True, exist_ok=True)
        manifest["languages"][language_name] = {"attributes": LANGUAGES[language_name]["attributes"], "files": {}}
        for split_offset, (split, count) in enumerate(split_counts.items()):
            records = generate_split(templates, language_name, split, count, args.seed + language_offset * 100 + split_offset)
            output_path = language_dir / f"{split}.jsonl"
            write_jsonl(output_path, records)
            manifest["languages"][language_name]["files"][split] = {"path": str(output_path), "records": len(records)}
            print(f"Wrote {len(records):,} {language_name} {split} records to {output_path}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
