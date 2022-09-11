# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.205.2/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
ARG VARIANT="3.9"
#FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}
FROM mcr.microsoft.com/vscode/devcontainers/anaconda:3

COPY requirements/* /tmp/pip-tmp/
RUN conda install -c conda-forge cvxpy
RUN pip3 install --upgrade pip
RUN pip3 --disable-pip-version-check --no-cache-dir install \
    -r /tmp/pip-tmp/requirements_dev.txt \
    -r /tmp/pip-tmp/requirements_docs.txt \
    && rm -rf /tmp/pip-tmp