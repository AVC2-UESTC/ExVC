## Documentation

## Artificial Dataset Preparation
ExVC used the **LibriTTS_R** dataset (train_clean-360 subset) and randomly selected recordings from 767 different speakers to create synthetic datasets. The list of speakers used in this process is attached. Additionally, we utilized **FreeVC** to convert the VCTK dataset using those 767 speakers.

### Synthetic Data Generation
1. **Download and Prepare LibriTTS_R**
   - Download the LibriTTS_R (train_clean-360) dataset from [here](https://www.openslr.org/resources/141/train_clean_360.tar.gz).
   - Unzip it and downsample it to **16kHz**.
   - Place the files in the `dataset` directory. The structure should look like:  
     `ExVC/dataset/LibriTTS_R/train-clean-360/xx/xx/xx.wav`.

2. **Download and Prepare VCTK**
   - Download the VCTK dataset from [here](https://datashare.ed.ac.uk/handle/10283/3443).
   - Unzip it, downsample it to **16kHz**, and place the files in your `dataset` directory.  
     Rename the folder to **DUMMY** to match the naming convention used in the filelists. The structure should look like:  
     `ExVC/dataset/DUMMY/xx/xx.wav`.

3. **Generate Synthetic Data**
   - Run the following command:  
     ```bash
     python CUDA_VISIBLE_DEVICES=0 python gen_art_dataset.py
     ```
   - The processed files will be saved to the `dataset/DUMMY_ART` folder.

### Metadata and Customization
- The file `artificial_dataset_generation.txt` is located in the `filelists` folder. Each line follows this format:  
  ```
  <path_to_save_generated_file>|<path_to_source_file>|<path_to_target_reference>
  ```
  - You can customize it based on your needs.

### Synthetic Data Details
- For each sample in the VCTK dataset, we generated **7 different synthetic audio samples** using roughly 7 reference speaker samples from the pool of 767 speakers in LibriTTS_R.
- The list of samples used is available. We used samples with durations ranging from **15 to 30 seconds** to ensure rich embeddings during conversion.
- The generated audio retains the semantic content of the original sample but adopts the timbre of the respective reference speaker.

---

## Training Input and Ground Truth

### Training Input
- Synthetic samples serve as the training input. Example:  
  ```
  DUMMY_ART/p318/p318_232__7555.wav
  ```
  - **Source Speaker**: `DUMMY_ART/p318/p318_232.wav` (from VCTK)
  - **Reference Speaker**: Speaker `7555` (from LibriTTS_R).

### Ground Truth Mapping
- To be more clear , if the training inputs are:
  ```
  DUMMY_ART/p318/p318_232__7555.wav
  DUMMY_ART/p318/p318_232__2269.wav
  ```
  Both inputs will use the same **ground truth** as the semantic information remains identical:  
  ```
  DUMMY_ART/p318/p318_232.wav
  ```
  The same speaker embedding will also be used.
   ```
   spk/p318/p318_232.npy
    ```
  - This can be understood as a **many-to-one mapping** since many different input will have the same ground truth.
---

## Preparation Steps

### Prepare the Speaker Embeddings
Run the following command:
```bash
python CUDA_VISIBLE_DEVICES=0 python preprocess_spk.py
```

### Prepare the SSL Features Using WavLM
Run the following command:
```bash
python CUDA_VISIBLE_DEVICES=0 python preprocess_ssl.py
```

### Launch the Training
To start training, run:
```bash
python CUDA_VISIBLE_DEVICES=0 python train.py
```

---

## Contact
For any issues, feel free to:
- Open an issue on GitHub.
- Reach out to me directly : 2672291403ATqqDOTcom 




