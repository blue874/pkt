#!/usr/bin/bash
conda env create -f env.yaml
conda activate flyp
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
