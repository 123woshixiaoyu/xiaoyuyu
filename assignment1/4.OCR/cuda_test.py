import torch, sys, importlib.util, os
print("torch file:", getattr(torch, "__file__", None))
print("torch version:", getattr(torch, "__version__", None))
print("cuda available:", hasattr(torch, "cuda") and torch.cuda.is_available())
print("sys.path[0]:", sys.path[0])
spec = importlib.util.find_spec("torch")
print("find_spec:", spec)
print("locations:", getattr(spec, "submodule_search_locations", None))
