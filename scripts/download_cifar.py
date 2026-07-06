"""Download CIFAR-10 to /tmp/cifar/ for use by the loader."""
import os
import urllib.request
import tarfile

DATA_DIR = "/tmp/cifar"
os.makedirs(DATA_DIR, exist_ok=True)

TGZ = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

if not os.path.exists(TGZ):
    print(f"downloading {URL} -> {TGZ}")
    urllib.request.urlretrieve(URL, TGZ)

if not os.path.exists(os.path.join(DATA_DIR, "cifar-10-batches-py")):
    print(f"extracting {TGZ}")
    with tarfile.open(TGZ, 'r:gz') as t:
        t.extractall(DATA_DIR)
    print("done")
else:
    print("already extracted")
