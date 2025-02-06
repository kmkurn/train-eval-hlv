# train-eval-hlv
Code for the paper "Training and Evaluating with Human Label Variation: An Empirical Study"

## Requirements

  1. Linux with CUDA 12.1
  1. Python 3.10+
  1. [pip-tools](https://github.com/jazzband/pip-tools)

Install dependencies with `pip-sync requirements.txt`.

If you have a non-Linux system or a different version of CUDA/Python, you may need to generate the appropriate `requirements.txt` yourself. See the [example command](https://github.com/kmkurn/train-eval-hlv/blob/9b4529093bb0492372cbf701860454de64677fc5/requirements.txt#L5) for reference.

## Tests

Ensure that everything works by running the tests with `pytest`. By default, slow tests (e.g., training models) are skipped. Run them by passing `-m slow` argument: `pytest -m slow`.

Also, tests in module `test_flair` are skipped by default. You may want to run them to verify the [FlairNLP](https://flairnlp.github.io/) version by passing `-k test_flair` argument: `pytest -k test_flair`. Combine with `-m slow` to run the slow tests too.

## Data

All Python scripts in this section can be invoked with `-h` or `--help` option to show their usage.

  1. Download the following datasets:
     - [HS-Brexit, MD-Agreement, and ArMIS](https://le-wi-di.github.io/)
     - [ChaosNLI](https://github.com/easonnie/ChaosNLI)
     - [MFRC](https://huggingface.co/datasets/USC-MOLA-Lab/MFRC)
  1. Convert the datasets into JSONL format using `convert_dataset.py`
  1. Create 10-fold train-test splits for ChaosNLI and MFRC using `create_kfold_splits.py`
  1. Create a dev set for each fold using `create_random_split.py`

## Training

```bash
./run_training.py train with chaosnli data_dir=data artifacts_dir=artifacts method=ReL
```

The above invocation trains a base RoBERTa on the ChaosNLI data under directory `data` using repeated labelling (ReL) method and saves the training artifacts (incl. the trained model parameters) under directory `artifacts`. Argument `train` and `chaosnli` are a *command* and a *named configuration* respectively, while key-value pairs such as `data_dir=data` are (unnamed) *configurations*. The `train` command can be omitted as it is the default command. The full list of commands can be viewed by executing the `help` command without any arguments.

The script also accepts extra configurations as a JSON file, which is useful for hyperparameter tuning with random search (see below).

## Random search

```bash
./generate_configs.py output_dir -n 20
```

The above invocation generates 20 JSON files under directory `output_dir`, each containing batch size and learning rate configurations sampled randomly. Invoke with `-h` or `--help` for detailed usage.

## Evaluation

The `run_training.py` script already performs evaluation after training completes. This section is mainly for evaluation with K-fold cross-validation (ChaosNLI and MFRC).

```bash
./run_eval.py with artifacts_dir=artifacts
```

The above invocation evaluates training artifacts under directory `artifacts`. For K-fold CV, artifacts `hlv.npy` and `label.dict` must be concatenated across folds beforehand.

## Annotation

NOTE: In this section, the term "HLV" refers to "human judgement distribution".

All Python scripts in this section can be invoked with `-h` or `--help` option to show their usage.

### Preprocessing

  1. Compute the mean HLV across runs for all models using `compute_method2hlv.py`
  1. Create the annotation input data using `create_hlv_annotation_input.py`

Annotators are expected to annotate the annotation input data. The expected format of the annotation output data is JSON whose schema can be viewed under directory `schemas`. The following instructions assume that the annotation output data exists.

### Postprocessing

  1. (Optional) Compute inter-annotator agreement using `compute_hlv_annotation_agreement.py`
  1. Compute vote distributions using `compute_vote_dist.py`

The result of the last instruction is a [DataFrame](https://pandas.pydata.org/docs/reference/frame.html#dataframe) serialised using [pickle](https://docs.python.org/3.10/library/pickle.html) with 5 columns:

  1. `model`, name of the pretrained model (e.g., `llama`);
  1. `method`, name of the HLV training method (e.g., `ReL`);
  1. `vote_dist`, a tuple of 4 numbers where each denotes the fraction of annotators who prefer:
      1. the predicted to the true HLV,
      1. both the predicted and the true HLV,
      1. the true to the predicted HLV, and
      1. neither the predicted nor the true HLV;
  1. `total`, number of annotators; and
  1. `nunique_texts`, number of unique input texts.

To get the fraction of annotators who prefer the predicted HLV, add the first and the second numbers of `vote_dist`.

## MongoDB integration with Sacred

Both `run_training.py` and `run_eval.py` scripts use [Sacred](https://pypi.org/project/sacred/) and have its `MongoObserver` activated by default. Set `SACRED_MONGO_URL` (and optionally `SACRED_DB_NAME`) environment variable(s) to write experiment runs to a MongoDB instance. For example, set `SACRED_MONGO_URL=mongodb://localhost:27017` if the MongoDB instance is listening on port 27017 on the local machine.

## License

Apache License, Version 2.0

## Citation

```
@misc{kurniawan2025,
  title = {Training and {{Evaluating}} with {{Human Label Variation}}: {{An Empirical Study}}},
  shorttitle = {Training and {{Evaluating}} with {{Human Label Variation}}},
  author = {Kurniawan, Kemal and Mistica, Meladel and Baldwin, Timothy and Lau, Jey Han},
  year = {2025},
  eprint = {2502.01891},
  doi = {10.48550/arXiv.2502.01891},
}
```

