import torch

print("PyTorch version:", torch.__version__)

# Sprawdź, czy jest wsparcie dla GPU (MPS = Metal Performance Shaders na Macu)
if torch.backends.mps.is_available():
    print("MPS (GPU) jest dostępne! Twój MacBook wykorzysta kartę graficzną.")
else:
    print("MPS (GPU) niedostępne – sprawdź instalację Pythona/PyTorch.")
