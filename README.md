# Data driven discretizations for solving 2D PDEs

This repository explores extensions of the techniques developed in:

>  [Learning data-driven discretizations for partial differential equations](https://www.pnas.org/content/116/31/15344).
  Yohai Bar-Sinai\*, Stephan Hoyer\*, Jason Hickey, Michael P. Brenner.
  PNAS 2019, 116 (31) 15344-15349.

See [this repository](https://github.com/google/data-driven-discretization-1d)
for the code used to produce results for the PNAS paper.

This is not an official Google product.

# VSCode Docker Container Development

1. [Install Docker on the host Ubuntu Machine](https://phoenixnap.com/kb/how-to-install-docker-on-ubuntu-18-04)

- Step 1: Update the system
  ```bash
    sudo apt-get update
  ```

- Step 2: Uninstall Old Versions of Docker
  ```bash
    sudo apt-get remove docker docker-engine docker.io
  ```
- Step 3: Install Docker
  ```bash
    sudo apt install docker.io
  ```
- Step 4: Start and Automate Docker
  ```bash
    sudo systemctl start docker

    sudo systemctl enable docker
  ```
- Step 5: (Optional) Check Docker version
  ```bash
    docker --version
  ```
2. [Install VSCode Editor on Ubuntu](https://code.visualstudio.com/download)

3. Open this folder in VSCode and allow for  `VSCode Remote` Extensions to be installed.

4. Running the Jupyter notebooks in the Docker cotnainer.

Running Jupypter notebook

```bash
    jupyter notebook --ip 0.0.0 --allow-root --no-browser
```

# Manual Installation

Installation is most easily done using pip.
1. Create or activate a virtual environment (e.g. using `virtualenv` or `conda`).
2. [Install TensorFlow](https://www.tensorflow.org/install/pip).
3. If you just want to install the package without the code,
   simply use pip to install directly from github:

   `pip install git+git//github.com/google-research/data-driven-pdes`

   If you want to fiddle around with the code, `cd` to where you want to store the code,
  clone the repo and install:
```bash
cd <your directory>
git clone git+https://github.com/google-research/data-driven-pdes
pip install -e data-driven-pdes
```

# Usage

We aim to make the code accessible for researchers who want to apply our method to their favorite PDEs. To this end we wrote, and continue to write, tutorials and documentation.
This is still very much in development, please [open an issue](https://github.com/google-research/data-driven-pdes/issues) if you have questions.

1. [A tutorial notebook](tutorial/Tutorial.ipynb) that explains some of the basic notions in the code base and demonstrates how to use the framework to define new equations.
2. [This notebook](tutorial/advection_1d.ipynb) contains a complete example of creating a training database, defining a model, training it and evaluating the trained model (well documented, though less pedagogical).
