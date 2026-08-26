# Workstation Environment Report

Validated on 2026-08-26 in the user-requested Conda environment `gnn311`.

| Component | Validated value |
|---|---|
| Hostname | `ustb` |
| OS | Ubuntu 22.04.5 LTS, Linux 6.8.0-136-generic x86_64 |
| CPU | Intel Core i7-14700, 28 logical CPUs reported |
| RAM | 31.1 GiB |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| VRAM | 8,188 MiB |
| NVIDIA driver | 575.64.03 |
| Driver CUDA compatibility | 12.9 |
| Local CUDA Toolkit | 11.7 (not used by the PyTorch wheel) |
| Python | 3.11.15 |
| Python executable | `/home/liulei/miniconda3/envs/gnn311/bin/python` |
| pip / setuptools / wheel | 26.2.1 / 78.1.0 / 0.48.0 |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| NumPy | 1.24.4 |
| Pandas | 2.0.3 |
| Matplotlib | 3.10.9 |
| pytest | 9.1.1 |
| OR-Tools | 9.11.4210 |
| Gurobi | 13.0.3 |
| psutil | 7.2.2 |

The official PyTorch CUDA 12.8 wheel was installed from
`https://download.pytorch.org/whl/cu128`. CUDA verification returned one
device, capability `(8, 9)`, and a successful 2048 by 2048 CUDA matrix
multiplication. `torch.cuda.is_available()` returned `True`.

Gurobi created and optimized a binary test model. The restricted non-production
license is valid through 2027-11-29.

The complete workstation package snapshot is recorded in
`requirements-lock.txt`. The shorter `requirements.txt` remains the formal
direct-dependency declaration.
