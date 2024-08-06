# XaiR
XaiR: An XR Platform that Integrates Large Language Models with the Physical World


 ## Setup Guide

Clone the repository onto your device:

```
git clone https://github.com/srutisrinidhi/XaiR.git
```

Clone apple's ferret multimodal LLM to a separate folder. Instructions can be found at [Apple ML-Ferret](https://github.com/apple/ml-ferret) 


You can use the virtual env already created by Ferret, or create your own if you do not plan to use ferret

To create your own:

```
cd XaiR
conda create -n XaiR python=3.10 -y
conda activate XaiR
```

Download essential libraries

```
conda install --yes --file requirements.txt
pip install --upgrade pip  # enable PEP 660 support
```

Generate certs:
```
mkdir ssl
openssl req -x509 -newkey rsa:2048 -keyout ssl/key.pem -out ssl/cert.pem -days 365
openssl rsa -in ssl/key.pem -out ssl/newkey.pem && mv ssl/newkey.pem ssl/key.pem
```

## Run Ferret
Enter the ferret directory and run
```
conda activate ferret
```

In a Terminal window, run
```
python -m ferret.serve.controller --host 0.0.0.0 --port 10000
```

In another Terminal window, run
```
CUDA_VISIBLE_DEVICES=0 python -m ferret.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path <PATH TO FERRET MODEL> --add_region_feature
```
