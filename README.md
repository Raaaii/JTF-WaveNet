
# JTF-WaveNet

JTF-WaveNet is dual-domain (time adn fequency) deep learning model designed for processing complex-valued NMR (Nuclear Magnetic Resonance) spectra. It employs a WaveNet-inspired architecture to reconstruct high-quality spectra from paired inputs (e.g., high-field and low-field spectra) while providing per-point uncertainty estimates.

## Overview

JTF-WaveNet addresses the challenge of reconstructing NMR spectra affected by artifacts such as vibrations or low signal-to-noise ratios. The model takes paired complex-valued spectra as input and outputs:
1. **Reconstructed Spectrum**: Artifacts corrected HRR spectrum.
2. **Uncertainty Estimates**: Per-point standard deviation (σ) values that quantify the model's confidence in each prediction.

The training process is divided into two stages to ensure stability and interpretability:
- **Stage 1**: Trains the reconstruction (mean) branch using Mean Squared Error (MSE).
- **Stage 2**: Trains the uncertainty branch while keeping the mean branch frozen, using uncertainty-aware loss functions.

This approach enables physically meaningful uncertainty quantification, which is crucial for scientific applications requiring reliable error bounds.

## Authors
- [@Rajka Pejanovic](https://www.github.com/raaaii)
- [@Lucas Siemons](https://www.github.com/l-siemons)


## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## Run Locally

Clone the project

```bash
git clone https://github.com/your-repo/JTF-WaveNet.git
```

Go to the project directory:

```bash
cd JTF-WaveNet
```
Create virtual environment adn activate it:
   ```bash
   # macOS/Linux
   python3.10 -m venv .venv
   source .venv/bin/activate

   # Windows
   py -3.10 -m venv .venv
   .venv\Scripts\activate
   ```

Upgrade pip and install build tools:
```bash
python -m pip install --upgrade pip setuptools wheel
```

Install the package in editable mode:

```bash
pip install -e .
```
Verify the installation:
   ```bash
   python -c "import jtf_wavenet; print('JTF-WaveNet imported successfully')"
   ```

For additional features, install optional dependencies:

```bash
# Development tools
pip install -e ".[dev]"

# Visualization enhancements
pip install -e ".[viz]"

# Machine learning utilities
pip install -e ".[ml]"

# Experiment tracking
pip install -e ".[tracking]"

# Web applications
pip install -e ".[apps]"
```
For Apple Silicon Macs, use the macOS-specific TensorFlow build:

```bash
pip install -e ".[mac]"
```


## Usage
### Input/Output Logic

**Input**:
- **Format**: Complex-valued numpy arrays or TensorFlow tensors
- **Shape**: `[batch_size, time_points, 2, 2]` where:
  - `time_points`: Number of spectral points (e.g., 4096)
  - First `2`: Real and imaginary components
  - Second `2`: Paired spectra (e.g., reference HF and HRR spectra)

**Output**:
- **Reconstructed Spectrum**: Shape `[batch_size, time_points, 2]` (real, imaginary)
- **Uncertainty**: Shape `[batch_size, time_points, 2]` (σ_real, σ_imaginary)

**Data Flow**:
1. Load paired NMR spectra - process spectra in NMRPipe or any other software
2. Preprocess (We need 2D arrays as input, NORMALIZATION)
3. Feed to model for inference
4. Post-process outputs (denormalize, apply IFFT (remember model's output is in frequncy domian))



**Some "good to know" infos**
We developed the testing scripts, all of them are in `scripts/` and can be used to visualise the singal, vibrations, to check the model shapes, hyperparametrs...! Again all of them are using `default_config.json` file but if you wish to evaluate your generator please define the path to your own `config.json`




# Experimental Data Prediction (JTF-WaveNet)

This folder contains a **fully working example pipeline** for running
**JTF-WaveNet on real experimental NMR data**.

The workflow is:
1. Process raw Bruker/NMRPipe data → phased FIDs
2. Organize experimental FIDs into a standard structure
3. Run neural-network prediction (`predict_all.py`)
4. Inspect results interactively (`plot_viewer.py`)

⚠️ Nothing in this folder is imported by `src/jtf_wavenet`.
This is **user-facing example code only**.

---

## Folder structure (recommended)

As it is organised in exp_data_predic!


## 1️⃣ Experimental data preparation (NMRPipe)

### Where this goes
Create a folder **outside `src/`**, e.g. exp_data_predic:

### Example NMRPipe processing script

To process the data using NMRpipe you have to write porcess.com file and generate fid.com `fid.com` or `process.com`, process file will look somehting like:

```csh
#!/bin/csh

nmrPipe -in test.fid \
| nmrPipe -fn EM -lb 0.3 \
| nmrPipe -fn ZF -auto \
| nmrPipe -fn FT -auto \
| nmrPipe -fn PS -p0 25.2 -p1 -56.4 -di -verb \
  -ov -out spectrum.ft

nmrPipe -in test.fid \
| nmrPipe -fn FT \
| nmrPipe -fn PS -p0 25.2 -p1 -56.4 -verb \
| nmrPipe -fn FT -inv \
  -ov -out fid_phased.fid
```
Now you will have to phase spectra manually and write down the p0, and p1 values and save it. To save it you will use `tcsh process.com` and once one spectrum is phased you can copy f`fid.com` or `process.com` outside (in this case in nmrpipe) directory. 
To run this inside all the experimental directories you can use `run_preprocess.sh` inside which you specify the root dir and output dir, as welll as NMRpipe dir: 
```bash
RAW_DIR="$RUN_DIR/data_raw"
PROC_DIR="$RUN_DIR/data_proc/fids_raw"
NMRPIPE_DIR="$RUN_DIR/nmrpipe"
```
To run `run_preprocess.sh` you can paste this in your terminal `bash run_preprocess.sh` but be carefull - you have to be in same directory.

## 2️⃣ Organize phased FIDs for the future use!
Once you have your phased FIDs saved in fids_raw, you can run the scirpt to organize them `organize_fids.py`, in the script you define 
```bash
source_root = "data_proc/fids_raw"
target_root = "data_proc/organized_fids"
```
## 3️⃣ Now we can use the model to predict!
Once youy have your npy arrays saved you can simply run `predic_all.py` script which will use `config.json` file to read importnat parameters and paths, like the directroy where your organized npy arrays are, the pm ref fid (any of the phased fids), the path to the chechpoints, to the high field spectrum and name of the FID inside all subfolder (it is fid_phased for all of them and this is defined in process.com), and many more - have a look! 👀:

🚨 ONE IMPORTANT THING: Data has to be normalize (it is done in `predic_all.py` script!! ):



```bash
  "ppm_ref_fid": "data_proc/organized_fids_300-329/vd300/fid_phased.fid",
  "checkpoint_dir": "../../checkpoints",

  "data_dir": "data_proc/organized_fids_300-329",
  "hf_file": "hf.fid",
  "fid_name": "fid_phased.fid",
  "vd_glob": "vd*",
```

## 4️⃣ Now we wanna see how our results look like!
This is easy! We have `plot_viewer.py` that is visualosation tool - by clicking n you are going to the next spectrum (next delay or next field)!


### Training a model (synthetic data)

In case you want to use the model and train on new data (chage some parameters, or even modify the signal fucntion)

Each training run lives in its own directory:

```bash
mkdir -p train/RUN_001
```
Expected structure:
```bash
train/RUN_001/
  config.json
  train.py
```

Prepare the generator config
```bash
cp configs/default_generator.json train/RUN_001/config.json
  ```
All synthetic data generation parameters are defined only in this JSON file:

spectrum – points, field strength, spectral width, echo times

peaks – number of peaks and all parameter distributions

vibration – vibration amplitudes, frequencies, relaxation

acquisition – noise, water peak, windowing, baseline (if enabled)

⚠️ No numeric parameters are hard-coded in src/.

### Run the training
```bash
python3 train/RUN_001/train.py
```
Be carefull - It will automatically run the `stage1` training but you can config it (easy, easy) to run  `stage2` by just changing it in main!

```bash
if __name__ == "__main__":
    # options: "stage1", "stage2"
    main(stage="stage2")
```

### Outputs

All outputs are written inside the run folder.
```bash
train/RUN_001/outputs/
  checkpoints/
  final_net_stage1.txt
  final_net_stage2.txt
  learning_rate_log.csv
```
Each run is fully self-contained.


### Use of evaluation script
We provide the `eval.py` script that you can find in `scripts/` that calls the model, reads the checkpoints, geenrates new data, and outputs the predictions (in case you want to see some examples). 
This script can be used for your own data - just define new checkpoints path, config path (where you define all values useed in your new geenrator)

Evaluate trained models using the evaluation script:

```bash
python scripts/eval_model.py \
    --stage stage2 \
    --binning soft \
    --ckpt checkpoints/stage2 \
    --output_dir runs/$(date +%Y-%m-%d_%H-%M-%S)
```

**Command-line Arguments**:
- `--stage`: Model stage to evaluate (`stage1` or `stage2`)
- `--binning`: Binning method for uncertainty calibration (`soft` or `hard`)
- `--ckpt`: Path to model checkpoint directory
- `--output_dir`: Directory to save evaluation results

There is list of arguments and defaults are all listed at the end of the `eval.py` script (HAVE A LOOK!!! 🫵)
## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
## Citation
If you use JTF-WaveNet in your research, please cite:

```bibtex
@software{jtf_wavenet,
  title = {Joint time-frequency WaveNet architecture for signal processing: application to vibration artifact suppression in high resolution relaxometry experiments},
  author = {R. Pejanovic, L. Siemons, A. Ruda, V. Thalakottoor, G. Bouvignies, P. Pelupessy, J. A.
Villanueva-Garibay, D. F. Hansen, and F. Ferrage},
  year = {2026},
  url = {https://github.com/Raaaii/JTF-WaveNet}
}
```