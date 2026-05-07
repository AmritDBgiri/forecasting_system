import os

# Force-load CUDA DLLs before TensorFlow sees them
_base = os.path.join(os.path.dirname(__file__), "venv", "Lib", "site-packages", "nvidia")

for _pkg in ["cudnn", "cublas", "cuda_nvrtc"]:
    _bin = os.path.join(_base, _pkg, "bin")
    if os.path.isdir(_bin):
        os.add_dll_directory(_bin)