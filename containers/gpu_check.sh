#!/bin/bash
set -u
echo "host: $(hostname)  start: $(date)"

python3 -c "
import torch
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
assert torch.cuda.get_device_capability(0) == (7, 5), 'not Turing -- wrong validation target'
" || exit 1

echo "end: $(date)"
