# How to Install TurtleWave hdEEG

This guide shows you how to install TurtleWave hdEEG and get it running on your system.

## Quick Install

The fastest way to install TurtleWave:

```bash
pip install turtlewave_hdEEG
```

This installs TurtleWave and all required dependencies.

!!! success "Installation complete?"
    Verify it worked:
    ```bash
    python -c "import turtlewave_hdEEG; print('Installation successful!')"
    ```

## System Requirements

**Python Version:** 3.8 or higher

**Operating Systems:** 
- macOS
- Linux
- Windows

**Disk Space:** ~500MB for package and dependencies

**Memory:** 8GB RAM minimum, 16GB recommended for large datasets

## Installation Methods

### Method 1: Install from PyPI (Recommended)

For most users, pip installation is the simplest approach:

```bash
pip install turtlewave_hdEEG
```

This automatically handles all dependencies.

### Method 2: Install from Source

If you need the latest development version or want to contribute:

```bash
git clone https://github.com/TancyKao/turtlewave-hdEEG.git
cd turtlewave-hdEEG
pip install -e .
```

The `-e` flag installs in editable mode, useful for development.

!!! tip "When to install from source"
    Use source installation if you:
    
    - Need features not yet in the PyPI release
    - Want to modify the code
    - Are contributing to development

### Method 3: Using Conda

If you prefer conda environments:

```bash
conda create -n turtlewave python=3.8
conda activate turtlewave
pip install turtlewave_hdEEG
```

!!! note "Why pip in conda?"
    TurtleWave isn't currently available as a conda package, but pip works fine within conda environments.

## Verify Your Installation

After installation, confirm everything works:

### Check Package Import

```bash
python -c "import turtlewave_hdEEG; print(turtlewave_hdEEG.__version__)"
```

You should see the version number printed.

### Launch the GUI

```bash
turtlewave_gui
```

The TurtleWave window should appear.

!!! success "GUI launched?"
    Perfect! Your installation is complete and working.

## Dependencies

TurtleWave automatically installs these required packages:

**Core Dependencies:**

- **PyQt5** - Graphical user interface
- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **scipy** - Scientific computing
- **h5py** - HDF5 file handling
- **matplotlib** - Visualization

**Analysis Dependencies:**

- **wonambi** - EEG analysis foundation
- **tensorpac** - Phase-amplitude coupling

All dependencies are installed automatically with TurtleWave.

## Troubleshooting

### PyQt5 Installation Issues

**Problem:** PyQt5 fails to install or GUI won't launch

**Solution:** Install PyQt5 separately first:

```bash
pip install PyQt5
```

On some systems, you may need system-level Qt libraries:

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-pyqt5
```

**macOS (using Homebrew):**
```bash
brew install pyqt5
```

**Windows:**
PyQt5 usually installs without issues. If problems occur, try:
```bash
pip install --upgrade pip
pip install PyQt5
```

### HDF5 Library Issues

**Problem:** h5py installation fails with HDF5 library errors

**Solution:** Install HDF5 development libraries:

**Ubuntu/Debian:**
```bash
sudo apt-get install libhdf5-dev
pip install h5py
```

**macOS (using Homebrew):**
```bash
brew install hdf5
pip install h5py
```

**Windows:**
Download pre-built wheels from [PyPI](https://pypi.org/project/h5py/#files)

### Wonambi Dependencies

**Problem:** Wonambi installation fails

**Solution:** Wonambi has its own dependencies. Check the [Wonambi documentation](https://wonambi-python.github.io/) for platform-specific requirements.

Common fixes:

```bash
# Update pip first
pip install --upgrade pip setuptools wheel

# Then try installing wonambi
pip install wonambi
```

### Permission Errors

**Problem:** "Permission denied" during installation

**Solution:** Use a virtual environment (recommended) or user installation:

```bash
pip install --user turtlewave_hdEEG
```

!!! warning "Avoid sudo pip"
    Don't use `sudo pip install` as it can cause system-wide conflicts. Use virtual environments instead.

### Import Errors After Installation

**Problem:** `ModuleNotFoundError` when importing

**Solution:** Ensure you're using the correct Python environment:

```bash
# Check which Python you're using
which python
python --version

# Check if package is installed
pip list | grep turtlewave
```

If installed in a different environment, activate that environment first.

## Virtual Environments (Recommended)

Using virtual environments prevents dependency conflicts and keeps your system clean.

### Using venv

```bash
# Create virtual environment
python -m venv turtlewave-env

# Activate it
source turtlewave-env/bin/activate  # macOS/Linux
turtlewave-env\Scripts\activate     # Windows

# Install TurtleWave
pip install turtlewave_hdEEG

# When done, deactivate
deactivate
```

### Using conda

```bash
# Create environment
conda create -n turtlewave python=3.8

# Activate it
conda activate turtlewave

# Install TurtleWave
pip install turtlewave_hdEEG

# When done, deactivate
conda deactivate
```

!!! tip "Why virtual environments?"
    Virtual environments:
    
    - Isolate project dependencies
    - Prevent version conflicts
    - Make it easy to reproduce your setup
    - Allow different Python versions for different projects

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade turtlewave_hdEEG
```

Check your current version:

```bash
python -c "import turtlewave_hdEEG; print(turtlewave_hdEEG.__version__)"
```

## Uninstalling

To remove TurtleWave:

```bash
pip uninstall turtlewave_hdEEG
```

This removes TurtleWave but keeps dependencies. To remove dependencies too:

```bash
pip uninstall turtlewave_hdEEG wonambi tensorpac
```

## Platform-Specific Notes

### macOS

- Works on both Intel and Apple Silicon (M1/M2) Macs
- May need Xcode Command Line Tools: `xcode-select --install`
- Homebrew useful for system dependencies

### Linux

- Tested on Ubuntu 20.04+ and similar distributions
- May need development packages: `build-essential`, `python3-dev`
- Check your distribution's package manager for dependencies

### Windows

- Works on Windows 10 and 11
- May need Visual C++ Build Tools for some dependencies
- PowerShell or Command Prompt both work

## Next Steps

Installation complete! Here's what to do next:

1. **Try the tutorial** - [Getting Started](../tutorials/getting-started.md) walks you through your first analysis
2. **Explore examples** - Check the `examples/` directory in the repository
3. **Read the docs** - Browse the [API reference](../reference/api/index.md) to understand capabilities

!!! question "Still having issues?"
    If you encounter problems not covered here:
    
    - Check [GitHub Issues](https://github.com/TancyKao/TurtleWave-hdEEG/issues) for similar problems
    - Open a new issue with your error message and system details
    - Include Python version, OS, and full error traceback

## Installation Checklist

Before moving on, verify:

- [ ] Python 3.8+ installed
- [ ] TurtleWave package installed (`pip list | grep turtlewave`)
- [ ] Can import package (`python -c "import turtlewave_hdEEG"`)
- [ ] GUI launches (`turtlewave_gui`)
- [ ] Using virtual environment (recommended)

If all checks pass, you're ready to start analyzing sleep data!