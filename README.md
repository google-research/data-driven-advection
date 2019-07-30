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
We aim to make the code accessible for researchers who want to apply our method to their favorite PDEs. To this end we wrote, and continue to write, tutorials and documentation.
This is still very much in development, please [open an issue](https://github.com/google-research/data-driven-pdes/issues) if you have questions. 

1. [A tutorial notebook](tutorial/Tutorial.ipynb) that explains some of the basic notions in the code base and demonstrates how to use the framework to define new equations.
2. [This notebook](tutorial/advection_1d.ipynb) contains a complete example of creating a training database, defining a model, training it and evaluating the trained model (well documented, though less pedagogical). 