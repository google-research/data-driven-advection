# Super-resolution methods for solving 2D PDEs

This is not an official Google product.

# Installation (MacOS/Linux)
This package requires protocol buffer to run, so installation is somewhat more involved than the usual python package. 
It is recomended that a virtual environment will be created (using either virtualenv or conda) and actuvated prior to installation.

```shell
curl -OL https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
unzip -o protoc-3.6.1-linux-x86_64.zip -d protoc3
pip install protobuf==3.6.1

git clone https://github.com/googleprivate/pde-superresolution-2d.git
SRC_DIR = pde-superresolution-2d/pde_superresolution_2d
protoc3/bin/protoc -I=$SRC_DIR --python_out=$SRC_DIR $SRC_DIR/metadata.proto
pip install -e pde-superresolution-2d
```
