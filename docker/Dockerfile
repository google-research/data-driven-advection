FROM continuumio/miniconda3

# core libraries
RUN pip install --no-cache-dir \
    tensorflow==1.14.0rc1 \
    absl-py apache-beam numpy xarray \
    scikit-learn matplotlib jupyterlab

# protobuf compiler
RUN conda install -y protobuf && \
    conda clean -afy
