# Copyright 2025 Kemal Kurniawan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

from flair.data import DataPair, Sentence
from flair.nn import Classifier

from corpus import NotNLICorpusError


# TODO change DataPair type to TextPair
def predict(artifacts_dir: Path, inputs_file: Path, outputs_file: Path, multilabel: bool = False) -> None:
    model = Classifier.load(artifacts_dir / "final-model.pt")
    lines = inputs_file.read_text("utf8").splitlines()
    dps: list[DataPair] | list[Sentence]
    is_nli = True
    try:
        dps = [_input2datapair(json.loads(l)) for l in lines]
    except NotNLICorpusError:
        dps = [_input2sent(json.loads(l)) for l in lines]
        is_nli = False
    model.predict(dps)
    label_type = "verdict" if is_nli else f"class{'es' if multilabel else ''}"
    with outputs_file.open("w", encoding="utf8") as f:
        for dp in dps:
            if is_nli or not multilabel:
                val = dp.get_label().value
                print(
                    json.dumps({label_type: val if is_nli else (val == "True")}), file=f
                )
            else:
                print(
                    json.dumps({label_type: [l.value for l in dp.get_labels()]}), file=f
                )


def _input2datapair(obj: dict, /) -> DataPair:
    try:
        text = obj["premise"]
    except KeyError:
        raise NotNLICorpusError
    return DataPair(Sentence(text), Sentence(obj["hypothesis"]))


def _input2sent(obj: dict, /) -> Sentence:
    return Sentence(obj["text"])
