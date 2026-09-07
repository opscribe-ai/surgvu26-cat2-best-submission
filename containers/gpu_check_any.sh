#!/bin/bash
set -u
echo "host: $(hostname)  start: $(date)"

python3 -c "
import numpy
import torch
assert torch.cuda.is_available(), 'no CUDA device visible inside the container'
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print('cuda True', name, 'capability', cap, '| numpy', numpy.__version__)
if cap != (7, 5):
    print('NOTE: this is NOT the Turing/sm_75 deployment target. This run proves '
          'the container works; it does NOT validate T4 behaviour. The pinned '
          'RTX 2080 Ti job is the check that does.')
" || exit 1

echo "end: $(date)"
