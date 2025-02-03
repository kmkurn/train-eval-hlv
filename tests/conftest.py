import json
from typing import Iterable

import pytest


@pytest.fixture
def write_nli_jsonl(tmp_path):
    def _write_jsonl(data: Iterable[str], filename: str) -> None:
        with (tmp_path / filename).open("w", encoding="utf8") as f:
            for dat in data:
                split = dat.split()
                p, h = split[:2]
                inp = {"premise": p, "hypothesis": h}
                obj: dict
                try:
                    v, a = split[2:]
                except ValueError:
                    try:
                        [v] = split[2:]
                    except ValueError:
                        obj = inp
                    else:
                        obj = {"input": inp, "output": {"verdict": v}}
                else:
                    obj = {
                        "input": inp,
                        "output": {"verdict": v},
                        "metadata": {"annotator": a},
                    }
                print(json.dumps(obj), file=f)

    return _write_jsonl


@pytest.fixture
def write_bc_jsonl(tmp_path):
    def _write_jsonl(data: Iterable[str], filename: str) -> None:
        with (tmp_path / filename).open("w", encoding="utf8") as f:
            for dat in data:
                split = dat.split()
                text = split[0]
                inp = {"text": text}
                obj: dict
                try:
                    c, a = split[1:]
                except ValueError:
                    try:
                        [c] = split[1:]
                    except ValueError:
                        obj = inp
                    else:
                        obj = {"input": inp, "output": {"class": bool(int(c))}}
                else:
                    obj = {
                        "input": inp,
                        "output": {"class": bool(int(c))},
                        "metadata": {"annotator": a},
                    }
                print(json.dumps(obj), file=f)

    return _write_jsonl


@pytest.fixture
def write_mullab_jsonl(tmp_path):
    def _write_jsonl(data: Iterable[str], filename: str) -> None:
        with (tmp_path / filename).open("w", encoding="utf8") as f:
            for dat in data:
                obj: dict
                if ";" not in dat:
                    obj = {"text": dat}
                else:
                    texts, annotator = dat.split(";")
                    split = texts.split()
                    inp = {"text": split[0]}
                    if split[1:]:
                        obj = {"input": inp, "output": {"classes": split[1:]}}
                    else:
                        obj = {"input": inp, "output": {"classes": []}}
                    if annotator:
                        obj["metadata"] = {"annotator": annotator}
                print(json.dumps(obj), file=f)

    return _write_jsonl
