# Multi-Stage-Diffusion-Augmentations

This project focuses on training a Multi-Stage-Diffusion-Augmentations model on various image segmentation datasets.

---

## Environment Installation

### Step 1: Install Miniconda

1. **Download Miniconda:**
   - Visit the [Miniconda Downloads Page](https://docs.conda.io/en/latest/miniconda.html).
   - Choose the installer for your operating system (Windows, macOS, or Linux).

2. **Install Miniconda:**
   - Follow the installation instructions for your OS:
     - **Windows:** Run the `.exe` installer and follow the prompts.
     - **macOS/Linux:** Open a terminal and run the `.sh` installer using:
       ```bash
       bash Miniconda3-latest-Linux-x86_64.sh
       ```
     - Agree to the license terms and choose installation settings.

3. **Verify Installation:**
   Open a terminal (or Command Prompt) and run:
   ```bash
   conda --version
   ```
   You should see the installed version of Conda.

4. **Create a New Environment (Optional):**
   To isolate dependencies for this project:
   ```bash
   conda create -n segmentation_env python=3.9
   conda activate segmentation_env
   ```

### Step 2: Install Project Dependencies

Once Miniconda is set up, install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Model Training

Run the following command for training:

```bash
nohup main.py > output.log 2>&1 &
```


## Environment Variables

The `example.env` file contains the required environment variables for running the project. Ensure to configure it properly before starting or use defaults.

---
