# See here for image contents: https://github.com/devcontainers/images/tree/main/src/anaconda

FROM mcr.microsoft.com/vscode/devcontainers/anaconda:latest

COPY requirements/* /tmp/pip-tmp/
RUN pip install cryptography==38.0.4
RUN conda install -c conda-forge cvxpy
RUN conda install -c conda-forge pandoc
RUN pip3 install --upgrade pip
RUN pip3 --disable-pip-version-check --no-cache-dir install \
    -r /tmp/pip-tmp/requirements_dev.txt \
    -r /tmp/pip-tmp/requirements_docs.txt \
    && rm -rf /tmp/pip-tmp