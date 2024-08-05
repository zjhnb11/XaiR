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
