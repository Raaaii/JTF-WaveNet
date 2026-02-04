# Experimental data prediction (NMRPipe → NPY → Predict → Plot)

This folder contains an example workflow to run JTF-WaveNet on real experimental data.


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

## 2️⃣ Organize phased FIDs for the future use!
Once you have your phased FIDs saved in fids_raw, you can run the scirpt to organize them `organize_fids.py`, in the script you define 
```bash
source_root = "data_proc/fids_raw"
target_root = "data_proc/organized_fids"
```
## 3️⃣ Now we can use the model to predict!
Once youy have your npy arrays saved you can simply run `predic_all.py` script which will use `config.json` file to read importnat parameters and paths, like the directroy where your organized npy arrays are, the pm ref fid (any of the phased fids), the path to the chechpoints, to the high field spectrum and name of the FID inside all subfolder (it is fid_phased for all of them and this is defined in process.com), and many more - have a look! 👀:

```bash
  "ppm_ref_fid": "data_proc/organized_fids_300-329/vd300/fid_phased.fid",
  "checkpoint_dir": "../../checkpoints",

  "data_dir": "data_proc/organized_fids_300-329",
  "hf_file": "hf.fid",
  "fid_name": "fid_phased.fid",
  "vd_glob": "vd*",
```

## 4️⃣ Now we wanna see how our results look like!
This is easy! We have `plot_viewer.py`
## Quick start

```bash
cd exp_data_predict/example_run

bash run_preprocess.sh
python3 organize_fids.py
python3 predict_all.py
python3 plot_viewer.py
```



```bash
# 1) Copy the example run
cp -r exp_data_predict/example_run exp_data_predict/MY_RUN
cd exp_data_predict/MY_RUN

# 2) Edit config
nano config.json

# 3) Preprocess with NMRPipe (creates data_proc/...)
bash run_preprocess.sh

# 4) Predict and save arrays
python3 predict_all.py

# 5) Interactive plotting (reads outputs/npy/)
python3 plot_viewer.py
