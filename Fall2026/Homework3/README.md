# Homework 3 — Multimodal Learning Assistant

**CAI 5607 · Fall 2026 · Q5: 45 points · Q6: 20 bonus points**

Read **Q5 and Q6 on Canvas** for the assignment, grading, survey, and submission requirements. This folder contains the starter implementation and the guides needed to use it. Begin with `hw3_submission.ipynb`.

## Setup

Download this whole folder. Use Python 3.10 or newer. From the `Homework3` directory:

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/check_setup.py
```

Open `hw3_submission.ipynb` in VS Code/Jupyter and select the `.venv` Python kernel. If you need a Jupyter UI, install `jupyterlab` separately and run `python -m jupyterlab`. The notebook starts in setup-only mode with live calls and downloads disabled. Complete the student functions and configure your models before enabling those switches.

## Q5 working files

- Complete the five TODOs in `student_agent.py` and the written responses in `hw3_submission.ipynb`.
- Build your corpus using `data/project/manifest.template.json` and [Source data](#source-data). Update the manifest path in `config.json`.
- Set exact model identifiers, retrieval settings, and execution limits in `config.json`; see [PROVIDERS.md](PROVIDERS.md). Keep credentials in environment variables or an ignored local `.env` based on `.env.example`.
- Install `requirements_embeddings.txt` for the required pretrained semantic retrieval comparison. Loading/downloading the model is enabled separately in the notebook.
- Create your development/test cases from `eval/case_template.json`; use [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for running and scoring the comparison.

The provided host modules (`hw3_utils.py`, `llm_backends.py`, `learning_session.py`, and `setup_helpers.py`) support the student agent. `experiment.py` runs the comparison; `summarize_run.py` summarizes the human-scored results. Keep these files with the notebook.

`data/demo/` contains six synthetic teaching documents and one toy chart for software checks. They do not satisfy the required student corpus. `DOC06` deliberately contains an untrusted instruction and a dummy token; treat it as test data. Evaluation fixtures live in `tests/`; create your own development and test cases in `eval/`.

## Source data

Copy `data/project/manifest.template.json` to `data/project/manifest.json`. Curate at least **10 substantive text documents and 5 visual items** about coherent GenAI topics. Duplicate files, extracted sections and individual chunks are not separate documents. The tiny demo collection does not count.

Store authorized UTF-8 `.md`/`.txt` extracts and actual PNG/JPEG/WebP visuals below `data/project/`. Use paths relative to the manifest. The loader reads only manifest-listed files and verifies SHA256. Keep originals or reproducible extraction instructions when redistribution is restricted. Add page/section markers to extracted text and keep original figure/page locators. Do not fill gaps with unlabeled generated text.

For every source record `id`, `title`, `kind`, `path`, `source` (original citation/URL), `permission`, `allow_external`, `date`, `version`, `status`, `locator`, `generated`, `content_origin`, `sha256`, and `derived_from` when applicable. Status may be `current`, `superseded` or `archived`. Retrieval defaults to current items. Record access dates in an additional `accessed_at` field when useful.

`derived_from` is a list of parent source IDs. A cropped figure should point to its document; a generated caption/summary should point to the underlying source and have `generated: true`. The supplied demo is standalone synthetic teaching material, explicitly labeled. A generated caption cannot replace actual pixel inspection. Preserve the source relationship even when you decide not to index derived text.

To compute a content hash without changing the file:

```bash
python scripts/hash_file.py data/project/text/SOURCE01.md
```

Set `config.json` → `manifest` to `data/project/manifest.json`. All configuration paths are relative to the Homework3 working directory. For hosted models, set `allow_external: true` only after checking permission to transmit each loaded source.

Keep development/test cases, gold answers, grading notes, learner attempts, Q6 logs and survey responses outside every retrieval manifest. The supporting source documents used to answer cases should remain indexed. Hashes detect changed bytes; they do not establish authenticity, licensing or substantive quality. Manually audit corpus counts and duplicate documents.

## Offline checks

The supplied infrastructure checks should pass before you implement the TODOs:

```bash
python -m unittest tests.test_provided tests.test_approval tests.test_learning_session tests.test_ui_support tests.test_assignment_requirements tests.test_reporting
```

Run the student contracts after completing the corresponding functions:

```bash
python -m unittest tests.test_student
python -m unittest tests.test_ui_student  # Optional Q6 Streamlit route
```

The student suites intentionally fail on the untouched starter. `unittest discover` includes them. Tests use scripted fixtures and mocked requests; they do not replace the real-model demonstrations or evaluation.

## Optional Q6 interface

The supplied Streamlit scaffold reuses Q5. Complete its four functions in `ui_student.py`; `ui_support.py` supplies the persistence and approval helpers used by `ui_app.py`.

```bash
python -m pip install -r requirements_ui.txt
# Copy config_q6.template.json to config_q6.json and configure it.
python -m streamlit run ui_app.py --server.address 127.0.0.1
```

Follow [UI_STARTER_GUIDE.md](UI_STARTER_GUIDE.md) for configuration and persistence. [CODE_GENERATION_SESSION_GUIDE.md](CODE_GENERATION_SESSION_GUIDE.md) provides the optional task card, reading links, and suggested learning sequence. Other web/desktop UI technologies are acceptable; follow Canvas for the Q6 report and survey.

## Generated work

Save actual logs, evaluation results, approved artifacts, Q6 progress, and screenshots under `outputs/`. Keep them out of retrieval indexes. Generated output files are ignored by Git; include the required, reviewed evidence in your Canvas submission. Exclude credentials, caches, and downloaded model weights.

Record your environment with `python -m pip freeze > outputs/environment_freeze.txt`. After saving the executed notebook, export it with `python -m nbconvert --to html hw3_submission.ipynb` (or provide the PDF export permitted by Canvas).
