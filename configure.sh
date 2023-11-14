#!/bin/bash

poetry install

gpu=$(lspci | grep -i '.* vga .* nvidia .*')
if [[ $gpu == *' NVIDIA '* ]]; then
    poetry run pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi
