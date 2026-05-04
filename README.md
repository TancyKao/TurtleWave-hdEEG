# TurtleWave hdEEG

High-density EEG processing for sleep research, extending [Wonambi](https://wonambi-python.github.io/) for large multi-channel datasets.

Detects sleep spindles, slow waves, K-complexes, and phase–amplitude coupling. Ships PyQt5 GUIs for both detection (`turtlewave_gui`) and event review (`eeg_review_gui`).

[Documentation](https://turtlewave-hdeeg.readthedocs.io/) · [Source](https://github.com/TancyKao/TurtleWave-hdEEG) · [Issues](https://github.com/TancyKao/TurtleWave-hdEEG/issues)

## Install (users)

```bash
pip install turtlewave-hdEEG
turtlewave_gui          # detection GUI
eeg_review_gui          # event review GUI
```

Requires Python ≥ 3.10. Tested on macOS and Linux.

## Develop (cloning the repo)

```bash
git clone https://github.com/TancyKao/TurtleWave-hdEEG.git
cd TurtleWave-hdEEG

# Create an isolated environment (Python ≥ 3.10)
python3 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies + the package in editable mode
pip install -r requirements.txt
pip install -e ".[dev]"

# Smoke test — should print the version with no errors
python -c "import turtlewave_hdEEG; print(turtlewave_hdEEG.__version__)"

# Optional: run the existing test script
python tests/test_turtlewave.py
```

`requirements.txt` is a fully-pinned lockfile generated from `pyproject.toml` via `uv pip compile`. To regenerate it after changing dependencies:

```bash
uv pip compile pyproject.toml --extra dev --extra docs --output-file requirements.txt
```

To work on the docs:

```bash
pip install -e ".[docs]"
mkdocs serve            # http://127.0.0.1:8000
```

## Repository layout

```
turtlewave_hdEEG/        # core library (detectors, processors, exporters)
frontend/                # PyQt5 GUIs
examples/                # standalone scripts and HPC batch templates
docs/                    # MkDocs Material source
tests/                   # smoke tests + synthetic fixtures
```

The HPC batch templates in `examples/NCI_commands/` use plain pip + venv on the cluster — no conda required.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use TurtleWave in your research, please cite the project:

```bibtex
@software{turtlewave,
  title  = {TurtleWave hdEEG: High-density EEG event detection for sleep research},
  author = {Kao, Tancy},
  url    = {https://github.com/TancyKao/TurtleWave-hdEEG}
}
```
