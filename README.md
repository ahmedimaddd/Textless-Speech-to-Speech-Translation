# Textless Speech-to-Speech Translation & Correction

This repository contains the code and findings for my graduation project, "Exploring Textless Speech-to-Speech: From Monolingual Correction to Cross-Lingual Translation." This project implements and validates a complete Speech-to-Unit Translation (S2UT) pipeline to perform complex speech transformation tasks without any intermediate text representation, based on the models and framework from Meta AI's Fairseq.

## Project Overview

This work explores the capabilities of textless speech-to-speech models, focusing on a novel application: **monolingual grammar correction**. The core of the project adapts the standard S2UT pipeline to correct grammatical errors in spoken English. This was achieved by pioneering a textless method to synthetically create paired data by programmatically corrupting the discrete acoustic units of clean speech.

## System Architecture

The system uses a three-stage architecture, as proposed in the S2UT framework:

1.  **Unit Extraction (Speech-to-Unit):** A pre-trained **HuBERT** model is used to extract high-level speech representations from the target audio. These representations are then clustered using **K-Means** to create a finite vocabulary of discrete "acoustic units."
2.  **Translation (Speech-to-Unit Translation):** A **Transformer**-based sequence-to-sequence model is trained to translate the source speech waveform directly into the sequence of target acoustic units.
3.  **Synthesis (Unit-to-Speech):** A pre-trained **HiFi-GAN** vocoder takes the predicted sequence of acoustic units and synthesizes them back into a high-fidelity speech waveform.

### Model Compatibility (Important)
For the pipeline to function correctly, the components must be compatible. Specifically, the HiFi-GAN vocoder must be trained on acoustic units generated with the **exact same configuration** as the units used in the translation task. This includes using the same **HuBERT layer** for feature extraction and the same **number of K-Means clusters**. Any mismatch will result in a noisy and unusable audio output. The models linked below are recommended as they are pre-configured for compatibility (100 clusters from HuBERT Base layer 6).

## Key Achievements

* **Pioneered a Novel Task:** Designed a fully textless method to create a synthetic paired dataset for **monolingual grammar correction**, achieving an impressive **ROUGE-L score of 85.10**.
* **Outperformed SOTA Baselines:** On the large-scale (128k samples) German-to-English translation task, the model achieved a **BLEU score of 28.21**, surpassing the published baseline scores from Meta AI's foundational S2UT paper.
* **Demonstrated Data-Scale Dependency:** Clearly showed through controlled experiments that the S2UT pipeline's performance is highly dependent on the scale of the training data.

## Required Models & Datasets

To replicate this project, you will need the specific pre-trained models and datasets used.

| Component | Description | Link |
| :--- | :--- | :--- |
| **K-Means Model** | 100-cluster model for HuBERT Base features (layer 6). | [km.bin](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin) |
| **Vocoder Checkpoint** | HiFi-GAN vocoder trained on LJSpeech. | [g_00500000](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/g_00500000) |
| **Vocoder Config** | Configuration file for the HiFi-GAN vocoder. | [config.json](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/config.json) |
| **LJSpeech Dataset** | Used for the grammar correction task. Paired data was created synthetically by corrupting the discrete units of the clean speech. | [Dataset Link](https://keithito.com/LJ-Speech-Dataset/) |
| **CoVoST 2 Dataset** | Source of German audio and English text for the large-scale translation task. | [Dataset Link](https://github.com/facebookresearch/covost) |
| **CVSS Corpus** | Source of target English audio, used for alignment with CoVoST 2. | [Dataset Link](https://github.com/google-research-datasets/cvss) |

## Technologies & Frameworks

* **Core Framework:** [Fairseq](https://github.com/facebookresearch/fairseq) (from Meta AI)
* **Deep Learning:** PyTorch, TorchAudio
* **Acoustic Unit Model:** [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) (as used in Fairseq)
* **Vocoder:** [Unit-based HiFi-GAN](https://github.com/facebookresearch/speech-resynthesis/tree/main/examples/speech_to_speech_translation) (trained with Speech Resynthesis repo)
* **Evaluation:** SacreBLEU, Librosa, ROUGE, OpenAI Whisper
* **Data Handling:** Pandas, NumPy, Scikit-learn

## Data Preparation

A significant part of this project involved a complex data preparation workflow.

* **Audio Format:** All source and target audio files **must be resampled to a 16kHz single-channel WAV format**. This is a strict requirement for the HuBERT model and the vocoder.
* **Unit Quantization:** For the target language, a manifest of all audio files is created. This is passed to a quantization script which uses HuBERT and K-Means to generate the discrete unit sequences.
* **Final Manifest:** The final training data is a tab-separated file (`.tsv`) where each line contains the path to a source audio file and the corresponding sequence of target unit IDs.

## Evaluation Methodology

Evaluating a textless system presents a unique challenge, as the model's output is audio, not text. To quantitatively measure performance, an ASR-based evaluation pipeline was designed:

1.  **Speech-to-Text Transcription:** The synthesized output audio from the model is transcribed back into text using a powerful ASR model. This project used **OpenAI's Whisper** for its high zero-shot accuracy.
2.  **Metric Calculation:** The transcribed text (hypothesis) is then compared against the ground-truth text (reference) using standard NLP metrics like BLEU and ROUGE to assess translation and correction quality.

## Setup & Installation

### Environment Recommendation (Important)
This project has specific dependencies that can cause issues in standard environments.
* **Google Colab:** Attempting to install `fairseq` on Colab will likely fail due to a version conflict. Colab uses Python 3.11 by default, while `fairseq` and its dependencies require **Python 3.10**.
* **Local Machine:** You may encounter similar library conflicts locally.

For these reasons, it is **highly recommended to use a cloud GPU platform like RunPod**, where you can create a dedicated environment using a **Python 3.10** container. This project was successfully developed and validated in such an environment.

### Installation Steps

1.  **Clone the Fairseq repository:**
    ```bash
    git clone [https://github.com/facebookresearch/fairseq.git](https://github.com/facebookresearch/fairseq.git)
    cd fairseq
    ```

2.  **Install Fairseq in editable mode:**
    ```bash
    pip install --editable ./
    ```

3.  **Install other required libraries:**
    Install the remaining packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

The project uses Fairseq's command-line tools for training and inference.

1.  **Training:**
    Training is launched using `fairseq-hydra-train`. An example command:
    ```bash
    fairseq-hydra-train \
      task.data=/path/to/your/data-manifest \
      --config-dir /path/to/config \
      --config-name config_s2ut.yaml
    ```

2.  **Inference (Generation):**
    To translate new audio files, use `fairseq-generate`:
    ```bash
    fairseq-generate /path/to/your/data-manifest \
      --path /path/to/your/trained_checkpoint.pt \
      --task speech_to_unit \
      --results-path /path/to/output_results
    ```

For detailed steps and configurations, please refer to the Jupyter Notebook (`Ahmedimad_GP_(3).ipynb`). For a comprehensive analysis of the methodology and results, please see the full thesis document included in this repository.
