# Super-resolution methods for solving 2D PDEs

This is not an official Google product.

# Installation
Installation is most easily done using pip.
1. Create or activate a virtual environment (e.g. using `virtualenv` or `conda`).
1. [Install TensorFlow](https://www.tensorflow.org/install/pip).
1. If you just want to install the package without the code,
   simply use pip to install directly from github:
   
   `pip install git+git//github.com/googleprivate/pde-superresolution-2d`
     
    If you want to fiddle around with the code, `cd` to where you want to store the code, 
    clone the repo and install: 
```bash
cd <your directory>
git clone git+https://github.com/googleprivate/pde-superresolution-2d
pip install -e pde-superresolution-2d
```

# Usage
We wrote a [tutorial notebook](Tutorial.ipynb) demonstrating how to use the framework to define new equations, create a training database, define a model and train it etc. This is still in development, please [open an issue](https://github.com/google-research/data-driven-pdes/issues) if you have questions. 