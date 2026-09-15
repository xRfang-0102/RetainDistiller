# RetainDistiller

Source implementation of **What Does Speech SSL Distillation Forget? Knowledge-Preserving Distillation of Speech Representations**, based on the supplied manuscript.

Includes wav2vec 2.0 Base, HuBERT Base and WavLM Base+ interfaces; two-layer students; CTC probe pretraining; MSE, posterior, feature and direct-CTC objectives; optional LEH; training/resume; evaluation; KL/CKA analysis; student export; and inference. No model weights, datasets or benchmark results are included. All model loading is local-only.

## Install

Use Python 3.10+ in an isolated environment. Install a [PyTorch build appropriate for your GPU](https://pytorch.org/get-started/locally/), then run from this directory:

```bash
python -m pip install -e .
```

## Download resources separately

| Resource | Download / access | Expected local location |
| --- | --- | --- |
| HuBERT Base | [Model files](https://huggingface.co/facebook/hubert-base-ls960/tree/main) | `checkpoints/hubert-base-ls960/` |
| wav2vec 2.0 Base | [Model files](https://huggingface.co/facebook/wav2vec2-base/tree/main) | `checkpoints/wav2vec2-base/` |
| WavLM Base+ | [Model files](https://huggingface.co/microsoft/wavlm-base-plus/tree/main) | `checkpoints/wavlm-base-plus/` |
| LibriSpeech | [OpenSLR](https://www.openslr.org/12) | `/path/to/LibriSpeech/` |
| CMU Pronouncing Dictionary | [Dictionary repository](https://github.com/cmusphinx/cmudict) | `/path/to/cmudict.dict` |
| VoxCeleb1 | [Dataset access](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) | User-defined |
| IEMOCAP | [Dataset access](https://sail.usc.edu/iemocap/) | User-defined |
| Speech Commands | [Dataset page](https://www.tensorflow.org/datasets/catalog/speech_commands) | User-defined |
| Fluent Speech Commands | [Dataset page](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) | User-defined |

Each teacher directory must contain its original `config.json` and compatible Hugging Face model weights (`model.safetensors`, `pytorch_model.bin`, or indexed shards). Raw Fairseq `.pt` checkpoints require separate conversion. No tokenizer or pretrained CTC head is required. 

## Quick start: HuBERT

Prepare the three local LibriSpeech subsets. These commands read existing audio and the local dictionary; they do not download anything.

```bash
retain-distiller prepare librispeech --root /path/to/LibriSpeech --split train-clean-100 --cmudict /path/to/cmudict.dict --output data/librispeech/train-clean-100.jsonl --vocab data/vocab/arpabet.json --oov skip
retain-distiller prepare librispeech --root /path/to/LibriSpeech --split dev-clean --cmudict /path/to/cmudict.dict --output data/librispeech/dev-clean.jsonl --oov skip
retain-distiller prepare librispeech --root /path/to/LibriSpeech --split test-clean --cmudict /path/to/cmudict.dict --output data/librispeech/test-clean.jsonl --oov skip
```

`--oov skip` drops complete utterances containing unknown words and writes a rejection report next to each manifest. Use identical resulting manifests for controlled comparisons. To retain the complete corpus, extend the local dictionary and use the default `--oov error`. Use `--text-only` without a dictionary for unlabeled distillation or ASR manifests.

Train the probe first, then the student:

```bash
retain-distiller probe --config configs/hubert.yaml
retain-distiller distill --config configs/hubert.yaml
retain-distiller export --checkpoint runs/hubert/posterior/last.pt --output exports/hubert-posterior
retain-distiller infer --model exports/hubert-posterior --audio /path/to/example.wav --output runs/example-features.pt
```

The probe checkpoint is resolved automatically as `runs/hubert/probe/best.pt`. Distillation exports only the student encoder; the teacher, probe, LEH and training heads are omitted. `infer` produces frame features, not text transcripts.

## Configuration and variants

All data/model paths are relative to the working directory; inherited YAML paths are relative to the containing YAML file. Replace `hubert.yaml` with `wav2vec2.yaml` or `wavlm.yaml` and pretrain a separate probe for each backbone.

```bash
retain-distiller show-config --config configs/hubert.yaml
retain-distiller distill --config configs/hubert.yaml --overlay configs/method/leh.yaml
retain-distiller distill --config configs/hubert.yaml --overlay configs/method/baseline.yaml
retain-distiller distill --config configs/hubert.yaml --overlay configs/method/feature.yaml
retain-distiller distill --config configs/hubert.yaml --overlay configs/method/direct_ctc.yaml
retain-distiller distill --config configs/hubert.yaml --resume runs/hubert/posterior/last.pt
```

| Overlay | Training objective |
| --- | --- |
| `posterior.yaml` | Representation MSE + weighted temperature-scaled posterior KL |
| `leh.yaml` | Posterior objective + weighted LEH reconstruction MSE |
| `baseline.yaml` | Representation MSE only; no probe required |
| `feature.yaml` | Representation MSE + frozen linguistic feature MSE |
| `direct_ctc.yaml` | Representation MSE + direct phoneme CTC; no pretrained probe required |

Override existing keys with `--set key=value ...`. The default effective batch size is 16: four utterances per microbatch and four accumulation steps. For less GPU memory, use `--set train.batch_size=1 train.accumulation_steps=16`. Probe and downstream batch settings live under `probe_train` and `downstream_train`. 
Custom weight interfaces are `backbone.teacher_path`, `paths.probe_checkpoint`, `downstream.upstream_path` and `--resume`. Use `paths.output_dir` for a separate run. Resume requires an unchanged resolved config and restores optimizer, scheduler, scaler, sampler position and random states.

## Evaluation

The built-in downstream runners are basic frozen-encoder evaluations, **not complete official SUPERB recipes**. PR/ASR use CTC with greedy decoding; KS/IC/SID/ER use masked mean pooling and classification. Train separate downstream heads for each teacher/student. Official splits, class definitions and decoding recipes must be supplied for benchmark comparisons; an S3PRL feature adapter is included.

```bash
retain-distiller downstream --config configs/hubert.yaml --overlay configs/task/pr.yaml
retain-distiller evaluate --checkpoint runs/hubert/posterior/pr/best.pt --manifest data/librispeech/test-clean.jsonl --output runs/hubert/pr-test.json
retain-distiller analyze --checkpoint runs/hubert/posterior/last.pt --manifest data/librispeech/test-clean.jsonl --output runs/hubert/knowledge.json
```

Use task overlays `asr`, `ks`, `ic`, `sid` or `er` for the other tasks. For ASR, first run `retain-distiller prepare characters --output data/vocab/characters.json`. For classification data, supply official split manifests and create a vocabulary from training labels with `retain-distiller prepare labels --manifest data/speech_commands/train.jsonl --output data/vocab/ks.json`.
Training writes `config.yaml`, `metrics.jsonl`, `last.pt` and `best.pt`. PER/WER/accuracy are fractions; retention is a percentage. `analyze` writes unscaled posterior KL and globally centered linear CKA. Use `retain-distiller retention --student STUDENT.json --teacher TEACHER.json --metric per --output RETENTION.json` for linguistic retention; use `--metric accuracy` for SID/ER.

## Source layout and checks

| Path | Purpose |
| --- | --- |
| `configs/` | Shared defaults, three backbones, five methods, six tasks |
| `src/retain_distiller/models/` | SSL wrappers, student initialization, probe, LEH, task heads |
| `src/retain_distiller/data.py`, `prepare.py` | Audio loading, manifests, local phonemization |
| `src/retain_distiller/system.py`, `engine.py` | Objectives, training, checkpointing, evaluation |
| `src/retain_distiller/analysis.py`, `metrics.py` | KL, CKA, PER/WER, accuracy, retention |
| `src/retain_distiller/inference.py` | Student-only export and local inference |
| `src/retain_distiller/s3prl_upstream.py` | Optional external evaluation adapter |
