# TurtleWave hdEEG

High-density EEG processing for sleep research, extending [Wonambi](https://wonambi-python.github.io/) for large multi-channel datasets. Detects sleep spindles, slow waves, K-complexes, and phase–amplitude coupling, and ships PyQt5 GUIs for both detection (`turtlewave_gui`) and event review (`eeg_review_gui`).

[Documentation](https://turtlewave-hdeeg.readthedocs.io/) · [Source](https://github.com/TancyKao/TurtleWave-hdEEG) · [Issues](https://github.com/TancyKao/TurtleWave-hdEEG/issues)

## Install

```bash
pip install turtlewave-hdEEG
```

Requires Python ≥ 3.10. Tested on macOS, Linux, and Windows.

Launch a GUI:

```bash
turtlewave_gui          # detection
eeg_review_gui          # event review
```

Or use the library directly:

```python
from turtlewave_hdEEG import LargeDataset, ParalSWA, CustomAnnotations

dataset = LargeDataset("subject001.set")
annot = CustomAnnotations("subject001.xml")

proc = ParalSWA(dataset=dataset, annotations=annot)
slow_waves = proc.detect_slow_waves(
    method="AASM/Massimini2004",
    chan=["Cz", "Fz"],
    stage=["NREM2", "NREM3"],
    json_dir="sw_results",
)
```

Full examples for spindles, K-complexes, and PAC are in [`examples/`](examples/) and the [documentation](https://turtlewave-hdeeg.readthedocs.io/).

## Develop

```bash
git clone https://github.com/TancyKao/TurtleWave-hdEEG.git
cd TurtleWave-hdEEG

python3 -m venv --prompt turtlewave .venv
source .venv/bin/activate           # macOS / Linux
# .venv\Scripts\activate            # Windows (PowerShell or cmd)

pip install -r requirements.txt
pip install -e ".[dev]"

python -c "import turtlewave_hdEEG; print(turtlewave_hdEEG.__version__)"
```

`requirements.txt` is a fully-pinned lockfile generated from `pyproject.toml` via `uv pip compile`. To regenerate after changing dependencies:

```bash
uv pip compile pyproject.toml --extra dev --extra docs --output-file requirements.txt
```

For docs work:

```bash
pip install -e ".[docs]"
mkdocs serve            # http://127.0.0.1:8000
```

## Repository layout

```
turtlewave_hdEEG/   # core library (detectors, processors, exporters)
frontend/           # PyQt5 GUIs
examples/           # standalone scripts and HPC batch templates
docs/               # MkDocs Material source
tests/              # smoke tests
```

HPC batch templates in `examples/NCI_commands/` target NCI Gadi (PBS) and use plain pip + venv on the cluster — no conda required.

## License

MIT — see [LICENSE](https://github.com/TancyKao/TurtleWave-hdEEG/blob/master/LICENSE).

## Citation

If you use TurtleWave in your research, please cite:

```bibtex
@software{turtlewave,
  title  = {TurtleWave hdEEG: High-density EEG event detection for sleep research},
  author = {Kao, Tancy},
  url    = {https://github.com/TancyKao/TurtleWave-hdEEG}
}
```
