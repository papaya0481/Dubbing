# Installing
Latest version has been updated with newer `numpy` and `transformers` lib to support wider range of hardware and models.

Run the following steps to install the required dependencies:

```base
conda create -n dubbing python=3.11 -y
conda activate dubbing
```

Install pytorch >= 2.8.0
```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Install MFA
```
conda install -c conda-forge montreal-forced-aligner
```

Install kaldi with cpu version
```
conda install -c conda-forge kaldi=*=*cpu*
```