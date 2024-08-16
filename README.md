# IPDM-PyTorch
Official implementation of paper  "Domain Progressive Low-dose CT Imaging using Iterative Partial Diffusion Model"

# IPDM usage guidance
First, please organize your datasets following the directory structure of the example dataset 
[URL].
```
E:
├─0.25dose
│  ├─image domain
│  │  ├─L067
│  │  │      L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA.npy
│  │  │      L067_FD_1_1.CT.0001.0011.2015.12.22.18.09.40.840353.358074459.IMA.npy
│  │  │      ............
│  │  └─L109
│  │          L109_FD_1_1.CT.0001.0001.2015.12.23.17.52.25.829117.125758448.IMA.npy
│  │          L109_FD_1_1.CT.0001.0011.2015.12.23.17.52.25.829117.125772038.IMA.npy
│  │          ............
│  └─projection domain
│      ├─L067
│      │      L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA.npy
│      │      L067_FD_1_1.CT.0001.0011.2015.12.22.18.09.40.840353.358074459.IMA.npy
│      │      ............
│      └─L109
│              L109_FD_1_1.CT.0001.0001.2015.12.23.17.52.25.829117.125758448.IMA.npy
│              L109_FD_1_1.CT.0001.0011.2015.12.23.17.52.25.829117.125772038.IMA.npy
│              ............
└─ND
    ├─image domain
    │  ├─L067
    │  │      L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA.npy
    │  │      L067_FD_1_1.CT.0001.0011.2015.12.22.18.09.40.840353.358074459.IMA.npy
    │  │      ............
    │  └─L109
    │          L109_FD_1_1.CT.0001.0001.2015.12.23.17.52.25.829117.125758448.IMA.npy
    │          L109_FD_1_1.CT.0001.0011.2015.12.23.17.52.25.829117.125772038.IMA.npy
    │          ............
    └─projection domain
        ├─L067
        │      L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA.npy
        │      L067_FD_1_1.CT.0001.0011.2015.12.22.18.09.40.840353.358074459.IMA.npy
        │      ............
        └─L109
                L109_FD_1_1.CT.0001.0001.2015.12.23.17.52.25.829117.125758448.IMA.npy
                L109_FD_1_1.CT.0001.0011.2015.12.23.17.52.25.829117.125772038.IMA.npy
                ............
```
### Runtime Environment: pytorch 1.7.1+cu110, libtorch 1.7.1+cu110, i9-13900K+RTX4090
### Test Samples
1. Download the example dataset from [URL] and store it in `Dataset/test sample`. Open `test_sample.ipynb`.
2. Run the code blocks in the jupyter notebook sequentially. To change the sample, simply modify the `idx=15`
in the second code block.


### Training on your own dataset
1. Prepare the training dataset, including: Image Domain NDCT + Projection Domain NDCT. 
Prepare the test dataset, including: Image Domain NDCT + Image Domain LDCT + Projection Domain NDCT + Projection Domain LDCT.
2. Modify the following code in `Dataset/npz_data_loader.py` according to the data file names to ensure 
that it can extract the case name and slice name.
```python    
def get_pname_sname(self, file_path):
        if self.patient_name is None:
            if self.data_type == "siemens":
                self.patient_name = list(map(lambda x: x.split("\\")[-2], file_path))
                self.slice_name = list(map(lambda x: x.split("\\")[-1].split(".")[0], file_path))
            elif self.data_type == "mayo":
                self.patient_name = list(map(lambda x: x.split("\\")[-2], file_path))
                self.slice_name = list(map(lambda x: x.split("\\")[-1].split(".")[-4], file_path))
```
#### Image Domain Training
3. Copy `Config/Mayo-Config/train_img_option.json` and modify the path parameters `"train_dataset_path_FD_img"`, `"train_dataset_path_FD_img"` 
to your own training data paths, and `"test_dataset_path_FD_img"`, `"test_dataset_path_LD_img"`, `"test_dataset_path_FD_proj"`, `"test_dataset_path_LD_proj"` to your own test data paths.
```json 
"data_type": "mayo",
"train_dataset_path_FD_img": "F:/ddpm_mayo/train/ND/image domain",
"train_dataset_path_LD_img": null,
"train_dataset_path_FD_img": "F:/ddpm_mayo/train/ND/projection domain",
"train_dataset_path_LD_proj": null,
"test_dataset_path_FD_img":  "F:/ddpm_mayo/test/ND/image domain",
"test_dataset_path_LD_img":  "F:/ddpm_mayo/test/0.25dose/image domain",
"test_dataset_path_FD_proj": "F:/ddpm_mayo/test/ND/projection domain",
"test_dataset_path_LD_proj": "F:/ddpm_mayo/test/0.25dose/projection domain",
```
4. Adjust the training and testing parameters in `Config/Mayo-Config/train_img_option.json`. 
It is recommended to use the default values. The meaning of each parameter is explained in detail in `Config/default_config.py`.
5. Open cmd, navigate to the directory containing `main.py`, and run:
```cmd
python main.py --load_option_path Config/Mayo-Config/train_img_option.json
```
6. To track training progress, find the log path and use TensorBoard by running in cmd:
```cmd
tensorboard --logdir Utils/ModelTrainLog/IPDM_train_Mayo_img/2024-08-15T16-54-23/trainSummary
```
#### Projection Domain Training
7. The process is almost the same as for Image Domain Training. Copy `Config/Mayo-Config/train_proj_option.json` and modify the path parameters and training/testing parameters.
8. Open cmd, navigate to the directory containing `main.py`, and run:
```cmd
python main.py --load_option_path Config/Mayo-Config/train_proj_option.json
```

### Testing on your own dataset
1. Copy `Config/Mayo-Config/test_progressive_option.json` and modify the test dataset paths to your own test data paths:
```json
    "test_dataset_path_FD_img": "Dataset/test sample/Mayo/ND/image domain",
    "test_dataset_path_LD_img": "Dataset/test sample/Mayo/0.25dose/image domain",
    "test_dataset_path_FD_proj": "Dataset/test sample/Mayo/ND/projection domain",
    "test_dataset_path_LD_proj": "Dataset/test sample/Mayo/0.25dose/projection domain",
```
2.Modify `load_img_model_path`, `load_proj_model_path` to the saved model parameters from your training, such as `Utils/ModelTrainLog/IPDM_train_Mayo_img/2024-08-15T16-54-23/save_models`. Modify `resume_epochs_proj`, `resume_epochs_img` to the checkpoint number you want to use.
3.Adjust the testing parameters in `Config/Mayo-Config/test_progressive_option.json`. It is recommended to use the default values. The smaller the `constant_guidance_img`, the more significant the denoising effect, but more details will be lost.
4.Open cmd, navigate to the directory containing `main.py`, and run
```cmd
python main.py --load_option_path Config/Mayo-Config/test_progressive_option.json
```

### Low-Dose Projection Simulation
First, set up the dynamic library runtime environment by downloading `libtorch1.7.1+cu110`, `pybind11`, and compile `Recon/TASART2DNSL0-Cpp`, or modify the following content to your own path:
`os.environ['path'] += ';E:/Liaofeiyang/libtorch-win-shared-with-deps-1.7.1+cu110/libtorch/lib;D:/Anaconda3/envs/diffusion;D:/Anaconda3/envs/diffusion/Lib/site-packages/torch/lib'`
1. Use the `proj_torch` function in the Tangential Projection Code `Recon/TASART2DNSL0.pyi` to obtain projection images.
2. Use `Utils/Low_dose_CT_simulate.py` to add noise to the clean projection images. The code for adding noise is as follows:
```python
import numpy as np
def add_noise(data: np.array, factor=0.25):
    Ne = 5.8
    N0 = 1.4 * 1e5
    n = np.random.randn(data.shape[0], data.shape[1])
    noise_data = data + np.sqrt(
        (1 - factor) * np.exp(data) * (1 + ((1 + factor) * Ne * np.exp(data)) / (factor * N0)) / (factor * N0)) * n
    return noise_data
```
3. Use the `recon_torch` function in the Iterative Reconstruction Code `Recon/TASART2DNSL0.pyi` to reconstruct noisy projections and obtain low-dose images.


