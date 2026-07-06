"""Download MNIST IDX files to /tmp/mnist/ for use by the loader."""
import os
import urllib.request

DATA_DIR = "/tmp/mnist"
os.makedirs(DATA_DIR, exist_ok=True)

FILES = [
    ("train-images-idx3-ubyte.gz",
     "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"),
    ("train-labels-idx1-ubyte.gz",
     "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"),
    ("t10k-images-idx3-ubyte.gz",
     "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"),
    ("t10k-labels-idx1-ubyte.gz",
     "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"),
]

for fname, url in FILES:
    out = os.path.join(DATA_DIR, fname.replace(".gz", ""))
    if os.path.exists(out):
        print(f"already exists: {out}")
        continue
    gz = os.path.join(DATA_DIR, fname)
    print(f"downloading {url} -> {gz}")
    urllib.request.urlretrieve(url, gz)
    import gzip, shutil
    with gzip.open(gz, 'rb') as f_in, open(out, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz)
    print(f"  decompressed to {out}")
