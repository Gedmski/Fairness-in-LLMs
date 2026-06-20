"""Dataset loading for local JSONL benchmark inputs."""

from __future__ import annotations

import json
import csv
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

from fair_mia.types import TextSample


def load_jsonl_samples(path: str | Path, *, is_member: bool, default_scenario: str) -> list[TextSample]:
    samples: list[TextSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sample_id = str(payload.get("sample_id") or f"{Path(path).stem}-{line_number}")
            text = str(payload["text"])
            group = str(payload["group"])
            scenario = str(payload.get("scenario", default_scenario))
            metadata = dict(payload.get("metadata", {}))
            attributes = {
                str(key): str(value)
                for key, value in dict(payload.get("attributes", {})).items()
                if value is not None
            }
            for key in ("gender", "language", "variety", "dataset"):
                if key not in attributes and metadata.get(key) is not None:
                    attributes[key] = str(metadata[key])
            samples.append(
                TextSample(
                    sample_id=sample_id,
                    text=text,
                    is_member=bool(payload.get("is_member", is_member)),
                    group=group,
                    scenario=scenario,
                    metadata=metadata,
                    attributes=attributes,
                )
            )
    return samples


def validate_binary_groups(samples: Iterable[TextSample]) -> None:
    groups = sorted({sample.group for sample in samples})
    if len(groups) != 2:
        raise ValueError(f"Expected exactly two groups, found {groups}.")


def validate_samples(samples: Iterable[TextSample]) -> None:
    seen_ids: set[str] = set()
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("Dataset must contain at least one sample.")
    for sample in sample_list:
        if not sample.sample_id:
            raise ValueError("sample_id must not be empty.")
        if sample.sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        if not sample.text.strip():
            raise ValueError(f"Sample {sample.sample_id} has empty text.")
        if sample.group not in {"G0", "G1"}:
            raise ValueError(f"Sample {sample.sample_id} has invalid group {sample.group!r}; expected G0 or G1.")
    validate_binary_groups(sample_list)


def load_benchmark_samples(members_path: str | Path, nonmembers_path: str | Path, *, default_scenario: str) -> list[TextSample]:
    members = load_jsonl_samples(members_path, is_member=True, default_scenario=default_scenario)
    nonmembers = load_jsonl_samples(nonmembers_path, is_member=False, default_scenario=default_scenario)
    samples = members + nonmembers
    validate_samples(samples)
    return samples


def write_jsonl_samples(samples: Iterable[TextSample], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = {
                "sample_id": sample.sample_id,
                "text": sample.text,
                "is_member": sample.is_member,
                "group": sample.group,
                "scenario": sample.scenario,
                "metadata": sample.metadata,
                "attributes": sample.attributes,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def prepare_pan_from_csv(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    text_field: str,
    group_field: str,
    split_field: str,
    member_split: str,
    id_field: str = "sample_id",
    group0_value: str | None = None,
    group1_value: str | None = None,
) -> tuple[Path, Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("PAN input CSV is empty.")
    for field in [text_field, group_field, split_field]:
        if field not in rows[0]:
            raise ValueError(f"Required field not found in CSV: {field}")

    observed_groups = sorted({row[group_field] for row in rows})
    if group0_value is None or group1_value is None:
        if len(observed_groups) != 2:
            raise ValueError("Provide --group0-value and --group1-value when the input has more or fewer than two group values.")
        group0_value, group1_value = observed_groups

    samples: list[TextSample] = []
    for index, row in enumerate(rows, start=1):
        group_raw = row[group_field]
        if group_raw == group0_value:
            group = "G0"
        elif group_raw == group1_value:
            group = "G1"
        else:
            continue
        sample_id = row.get(id_field) or f"pan-{index}"
        samples.append(
            TextSample(
                sample_id=str(sample_id),
                text=row[text_field],
                is_member=row[split_field] == member_split,
                group=group,
                scenario="finetuning",
                metadata={"source": "pan", "original_group": group_raw, "split": row[split_field]},
                attributes={
                    "gender": str(group_raw).lower(),
                    "dataset": "pan_csv",
                },
            )
        )
    validate_samples(samples)
    members_path = output_dir / "members.jsonl"
    nonmembers_path = output_dir / "nonmembers.jsonl"
    write_jsonl_samples([sample for sample in samples if sample.is_member], members_path)
    write_jsonl_samples([sample for sample in samples if not sample.is_member], nonmembers_path)
    return members_path, nonmembers_path


def prepare_pan17_from_xml(
    *,
    train_dir: str | Path,
    test_dir: str | Path,
    output_dir: str | Path,
    lang: str = "en",
    tokenizer_model_id: str = "EleutherAI/pythia-160m",
    cache_dir: str | Path | None = None,
    window_tokens: int = 256,
    max_windows_per_bucket: int = 250,
    seed: int = 0,
    balance: bool = True,
) -> tuple[Path, Path]:
    if window_tokens <= 0:
        raise ValueError("window_tokens must be positive.")
    if max_windows_per_bucket <= 0:
        raise ValueError("max_windows_per_bucket must be positive.")

    train_lang_dir = _resolve_pan17_lang_dir(train_dir, lang)
    test_lang_dir = _resolve_pan17_lang_dir(test_dir, lang)
    train_truth = _load_pan17_truth(train_lang_dir / "truth.txt")
    test_truth = _load_pan17_truth(test_lang_dir / "truth.txt")
    tokenizer = _load_pan17_tokenizer(tokenizer_model_id, cache_dir)

    members = _build_pan17_samples(
        train_lang_dir,
        train_truth,
        tokenizer=tokenizer,
        tokenizer_model_id=tokenizer_model_id,
        lang=lang,
        window_tokens=window_tokens,
        is_member=True,
    )
    nonmembers = _build_pan17_samples(
        test_lang_dir,
        test_truth,
        tokenizer=tokenizer,
        tokenizer_model_id=tokenizer_model_id,
        lang=lang,
        window_tokens=window_tokens,
        is_member=False,
    )
    if balance:
        members, nonmembers = _balance_pan17_samples(
            members,
            nonmembers,
            max_windows_per_bucket=max_windows_per_bucket,
            seed=seed,
        )
    samples = members + nonmembers
    validate_samples(samples)

    output_dir = Path(output_dir)
    members_path = output_dir / "members.jsonl"
    nonmembers_path = output_dir / "nonmembers.jsonl"
    write_jsonl_samples(members, members_path)
    write_jsonl_samples(nonmembers, nonmembers_path)
    return members_path, nonmembers_path


def _load_pan17_tokenizer(tokenizer_model_id: str, cache_dir: str | Path | None) -> object:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install research dependencies before preparing PAN 2017 token windows: pip install -e .[research]"
        ) from exc

    kwargs = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return AutoTokenizer.from_pretrained(tokenizer_model_id, **kwargs)


def _resolve_pan17_lang_dir(base_dir: str | Path, lang: str) -> Path:
    base_path = Path(base_dir)
    lang_path = base_path / lang
    candidate = lang_path if lang_path.exists() else base_path
    if not candidate.exists():
        raise FileNotFoundError(f"PAN 2017 directory not found: {base_path}")
    if not (candidate / "truth.txt").exists():
        raise FileNotFoundError(f"PAN 2017 truth file not found under: {candidate}")
    return candidate


def _load_pan17_truth(path: str | Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split(":::")
            if len(parts) != 3:
                raise ValueError(f"Invalid PAN 2017 truth row at {path}:{line_number}")
            author_id, gender, variety = parts
            labels[author_id] = {"gender": gender.strip().lower(), "variety": variety.strip().lower()}
    return labels


def _build_pan17_samples(
    directory: Path,
    truth: dict[str, dict[str, str]],
    *,
    tokenizer: object,
    tokenizer_model_id: str,
    lang: str,
    window_tokens: int,
    is_member: bool,
) -> list[TextSample]:
    samples: list[TextSample] = []
    for xml_path in sorted(directory.glob("*.xml")):
        author_id = xml_path.stem
        if author_id not in truth:
            continue
        gender = truth[author_id]["gender"]
        if gender == "female":
            group = "G0"
        elif gender == "male":
            group = "G1"
        else:
            continue
        text = _read_pan17_author_text(xml_path)
        samples.extend(
            _window_pan17_author_text(
                author_id=author_id,
                text=text,
                group=group,
                original_group=gender,
                variety=truth[author_id]["variety"],
                split="train" if is_member else "test",
                lang=lang,
                tokenizer=tokenizer,
                tokenizer_model_id=tokenizer_model_id,
                window_tokens=window_tokens,
                is_member=is_member,
            )
        )
    return samples


def _read_pan17_author_text(path: str | Path) -> str:
    root = ET.parse(path).getroot()
    documents = [document.text.strip() for document in root.findall(".//document") if document.text and document.text.strip()]
    if not documents:
        raise ValueError(f"PAN 2017 XML has no non-empty documents: {path}")
    return "\n".join(documents)


def _window_pan17_author_text(
    *,
    author_id: str,
    text: str,
    group: str,
    original_group: str,
    variety: str,
    split: str,
    lang: str,
    tokenizer: object,
    tokenizer_model_id: str,
    window_tokens: int,
    is_member: bool,
) -> list[TextSample]:
    encode = getattr(tokenizer, "encode", None)
    decode = getattr(tokenizer, "decode", None)
    if encode is None or decode is None:
        raise ValueError("PAN 2017 tokenizer must provide encode() and decode().")

    token_ids = list(encode(text, add_special_tokens=False))
    samples: list[TextSample] = []
    full_windows = len(token_ids) // window_tokens
    for window_index in range(full_windows):
        start = window_index * window_tokens
        stop = start + window_tokens
        window_ids = token_ids[start:stop]
        window_text = str(decode(window_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)).strip()
        if not window_text:
            continue
        samples.append(
            TextSample(
                sample_id=f"{lang}:{split}:{author_id}:w{window_index}",
                text=window_text,
                is_member=is_member,
                group=group,
                scenario="finetuning",
                metadata={
                    "author_id": f"{lang}:{split}:{author_id}",
                    "original_author_id": author_id,
                    "window_index": window_index,
                    "token_count": len(window_ids),
                    "original_group": original_group,
                    "variety": variety,
                    "split": split,
                    "source": "pan17",
                    "lang": lang,
                    "tokenizer_model_id": tokenizer_model_id,
                },
                attributes={
                    "gender": original_group,
                    "language": lang,
                    "variety": variety,
                    "dataset": "pan17",
                },
            )
        )
    return samples


def _balance_pan17_samples(
    members: list[TextSample],
    nonmembers: list[TextSample],
    *,
    max_windows_per_bucket: int,
    seed: int,
) -> tuple[list[TextSample], list[TextSample]]:
    buckets: dict[tuple[str, str, bool], list[TextSample]] = defaultdict(list)
    for sample in members + nonmembers:
        variety = str(sample.metadata.get("variety", "unknown"))
        buckets[(sample.group, variety, sample.is_member)].append(sample)

    rng = random.Random(seed)
    balanced_members: list[TextSample] = []
    balanced_nonmembers: list[TextSample] = []
    surviving_varieties = sorted({variety for _, variety, _ in buckets})
    for variety in surviving_varieties:
        required_keys = [
            ("G0", variety, True),
            ("G0", variety, False),
            ("G1", variety, True),
            ("G1", variety, False),
        ]
        if any(not buckets[key] for key in required_keys):
            continue
        limit = min(len(buckets[key]) for key in required_keys)
        limit = min(limit, max_windows_per_bucket)
        if limit <= 0:
            continue
        for key in required_keys:
            candidates = list(buckets[key])
            rng.shuffle(candidates)
            selected = candidates[:limit]
            if key[2]:
                balanced_members.extend(selected)
            else:
                balanced_nonmembers.extend(selected)

    if not balanced_members or not balanced_nonmembers:
        raise ValueError(
            "PAN 2017 matching yielded no balanced group x variety x membership buckets with usable token windows."
        )

    return balanced_members, balanced_nonmembers


def prepare_pan17_multilingual(
    *,
    train_dir: str | Path,
    test_dir: str | Path,
    output_dir: str | Path,
    languages: tuple[str, ...] = ("en", "es", "pt", "ar"),
    tokenizer_model_id: str = "Qwen/Qwen3-4B-Base",
    cache_dir: str | Path | None = None,
    window_tokens: int = 256,
) -> tuple[Path, Path]:
    tokenizer = _load_pan17_tokenizer(tokenizer_model_id, cache_dir)
    members: list[TextSample] = []
    nonmembers: list[TextSample] = []
    for lang in languages:
        train_lang_dir = _resolve_pan17_lang_dir(train_dir, lang)
        test_lang_dir = _resolve_pan17_lang_dir(test_dir, lang)
        members.extend(
            _build_pan17_samples(
                train_lang_dir,
                _load_pan17_truth(train_lang_dir / "truth.txt"),
                tokenizer=tokenizer,
                tokenizer_model_id=tokenizer_model_id,
                lang=lang,
                window_tokens=window_tokens,
                is_member=True,
            )
        )
        nonmembers.extend(
            _build_pan17_samples(
                test_lang_dir,
                _load_pan17_truth(test_lang_dir / "truth.txt"),
                tokenizer=tokenizer,
                tokenizer_model_id=tokenizer_model_id,
                lang=lang,
                window_tokens=window_tokens,
                is_member=False,
            )
        )
    validate_samples(members + nonmembers)
    output_dir = Path(output_dir)
    members_path = output_dir / "members.jsonl"
    nonmembers_path = output_dir / "nonmembers.jsonl"
    write_jsonl_samples(members, members_path)
    write_jsonl_samples(nonmembers, nonmembers_path)
    return members_path, nonmembers_path


def prepare_pan18_from_xml(
    *,
    train_dir: str | Path,
    test_dir: str | Path,
    output_dir: str | Path,
    languages: tuple[str, ...] = ("en", "es", "ar"),
    tokenizer_model_id: str = "Qwen/Qwen3-4B-Base",
    cache_dir: str | Path | None = None,
    window_tokens: int = 256,
) -> tuple[Path, Path]:
    tokenizer = _load_pan17_tokenizer(tokenizer_model_id, cache_dir)
    members: list[TextSample] = []
    nonmembers: list[TextSample] = []
    for lang in languages:
        for base_dir, is_member in ((train_dir, True), (test_dir, False)):
            lang_dir = _resolve_pan17_lang_dir(base_dir, lang)
            truth = _load_pan18_truth(lang_dir / "truth.txt")
            for xml_path in sorted(lang_dir.glob("*.xml")):
                author_id = xml_path.stem
                gender = truth.get(author_id)
                if gender not in {"female", "male"}:
                    continue
                group = "G0" if gender == "female" else "G1"
                windows = _window_pan17_author_text(
                    author_id=author_id,
                    text=_read_pan17_author_text(xml_path),
                    group=group,
                    original_group=gender,
                    variety="unknown",
                    split="train" if is_member else "test",
                    lang=lang,
                    tokenizer=tokenizer,
                    tokenizer_model_id=tokenizer_model_id,
                    window_tokens=window_tokens,
                    is_member=is_member,
                )
                windows = [
                    replace(
                        sample,
                        sample_id=sample.sample_id.replace(f"{lang}:", f"pan18:{lang}:", 1),
                        metadata={**sample.metadata, "source": "pan18"},
                        attributes={**sample.attributes, "dataset": "pan18"},
                    )
                    for sample in windows
                ]
                (members if is_member else nonmembers).extend(windows)
    validate_samples(members + nonmembers)
    output_dir = Path(output_dir)
    members_path = output_dir / "members.jsonl"
    nonmembers_path = output_dir / "nonmembers.jsonl"
    write_jsonl_samples(members, members_path)
    write_jsonl_samples(nonmembers, nonmembers_path)
    return members_path, nonmembers_path


def _load_pan18_truth(path: str | Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = [part.strip().lower() for part in line.strip().split(":::")]
            if len(parts) < 2:
                raise ValueError(f"Invalid PAN 2018 truth row at {path}:{line_number}")
            labels[parts[0]] = parts[1]
    return labels


def prepare_pile_sample(
    *,
    output_dir: str | Path,
    max_members: int,
    max_nonmembers: int,
    cache_dir: str | Path | None = None,
    subset: str | None = None,
) -> tuple[Path, Path]:
    try:
        import datasets
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install research dependencies before preparing Pile samples: pip install -e .[research]") from exc

    major_version = int(str(getattr(datasets, "__version__", "0")).split(".", 1)[0])
    if major_version >= 5:
        raise RuntimeError(
            "prepare-pile-sample requires a datasets version below 5 because EleutherAI/pile still uses a dataset script. "
            "Install a compatible release, for example: pip install 'datasets<5'"
        )

    dataset_kwargs = {
        "path": "EleutherAI/pile",
        "split": "train",
        "streaming": True,
    }
    if subset:
        dataset_kwargs["name"] = subset
    if cache_dir:
        dataset_kwargs["cache_dir"] = str(cache_dir)
    stream = load_dataset(**dataset_kwargs)
    members: list[TextSample] = []
    nonmembers: list[TextSample] = []
    for index, row in enumerate(stream):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        target = members if len(members) < max_members else nonmembers
        if len(nonmembers) >= max_nonmembers:
            break
        sample = TextSample(
            sample_id=f"pile-{index}",
            text=text[:4000],
            is_member=target is members,
            group="G0" if index % 2 == 0 else "G1",
            scenario="pretraining",
            metadata={"source": "pile", "synthetic_group_assignment": True},
            attributes={"dataset": "pile"},
        )
        target.append(sample)
    samples = members + nonmembers
    validate_samples(samples)
    output_dir = Path(output_dir)
    members_path = output_dir / "members.jsonl"
    nonmembers_path = output_dir / "nonmembers.jsonl"
    write_jsonl_samples(members, members_path)
    write_jsonl_samples(nonmembers, nonmembers_path)
    return members_path, nonmembers_path


def summarize_samples(samples: Iterable[TextSample]) -> dict[str, object]:
    sample_list = list(samples)
    by_group = Counter(sample.group for sample in sample_list)
    by_membership = Counter("member" if sample.is_member else "nonmember" for sample in sample_list)
    by_group_membership = Counter(
        f"{sample.group}:{'member' if sample.is_member else 'nonmember'}" for sample in sample_list
    )
    varieties = {
        str(sample.attributes.get("variety") or sample.metadata.get("variety", "")).strip().lower()
        for sample in sample_list
        if str(sample.attributes.get("variety") or sample.metadata.get("variety", "")).strip()
    }
    by_group_variety_membership = Counter(
        f"{sample.group}:{str(sample.attributes.get('variety') or sample.metadata.get('variety', 'unknown')).strip().lower()}:"
        f"{'member' if sample.is_member else 'nonmember'}"
        for sample in sample_list
        if str(sample.attributes.get("variety") or sample.metadata.get("variety", "")).strip()
    )
    group_membership_counts = [count for _, count in sorted(by_group_membership.items())]
    summary = {
        "total": len(sample_list),
        "groups": dict(sorted(by_group.items())),
        "membership": dict(sorted(by_membership.items())),
        "group_membership": dict(sorted(by_group_membership.items())),
        "is_four_cell_balanced": bool(group_membership_counts) and len(set(group_membership_counts)) == 1,
    }
    if varieties:
        summary["varieties"] = sorted(varieties)
        summary["group_variety_membership"] = dict(sorted(by_group_variety_membership.items()))
    languages = Counter(
        str(sample.attributes.get("language") or sample.metadata.get("lang", "unknown"))
        for sample in sample_list
    )
    if languages and set(languages) != {"unknown"}:
        summary["languages"] = dict(sorted(languages.items()))
    authors = {
        str(sample.attributes.get("author_id") or sample.metadata.get("author_id"))
        for sample in sample_list
        if sample.attributes.get("author_id") or sample.metadata.get("author_id")
    }
    if authors:
        summary["authors"] = len(authors)
    return summary
