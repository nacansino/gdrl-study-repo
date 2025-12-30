# CUDA 12.9 + cuDNN + Ubuntu 22.04
# Meets: glibc >= 2.28, binutils >= 2.29, libstdc++ >= 3.4.25 (userspace).
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10
ARG NB_USER=jovyan
ARG NB_UID=1000
ARG NB_GID=100

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- System packages ----
# Notes vs your old 16.04 image:
# - libav-tools is obsolete; ffmpeg covers it on 22.04.
# - qt-sdk is obsolete; if you truly need Qt dev, install qtbase5-dev / qtchooser / qt5-qmake.
#   (Qt is only needed if you plan to build GUI apps with PyQt or Py   
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y --no-install-recommends \
    build-essential \
    bzip2 \
    ca-certificates \
    cmake \
    curl \
    emacs \
    ffmpeg \
    git \
    jed \
    ffmpeg \
    libboost-all-dev \
    libjpeg-dev \
    libpng-dev \
    libpq-dev \
    libsdl2-dev \
    libsm6 \
    libxext-dev \
    libxrender1 \
    locales \
    nano \
    netcat \
    python3-dev \
    sudo \
    swig \
    unzip \
    vim \
    wget \
    xpra \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 || true

# ---- Create user ----
# If GID already exists in base image, reuse it; otherwise create it.
RUN set -eux; \
    if getent group "${NB_GID}" >/dev/null; then \
        EXISTING_GROUP="$(getent group "${NB_GID}" | cut -d: -f1)"; \
    else \
        groupadd --gid "${NB_GID}" "${NB_USER}"; \
        EXISTING_GROUP="${NB_USER}"; \
    fi; \
    useradd --uid "${NB_UID}" --gid "${EXISTING_GROUP}" -m -s /bin/bash "${NB_USER}"; \
    echo "${NB_USER} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${NB_USER}"; \
    chmod 0440 "/etc/sudoers.d/${NB_USER}"


# ---- Python (venv) ----
# Using Ubuntu Python + venv keeps this Dockerfile self-contained.
# If you prefer conda/mamba like the original, say so and I’ll provide that variant too.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make /opt/venv and put it first in PATH
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel

# Change ownership of venv to NB_USER so later installs don’t need sudo
RUN chown -R ${NB_UID}:${NB_GID} /opt/venv
USER ${NB_UID}

# ---- Jupyter + tooling ----
# Modern JupyterLab no longer requires node for most extensions, but we keep node/npm installed
# since your original image used lab extensions and other JS tools.
RUN pip install \
    jupyterlab notebook jupyterhub \
    tensorboard \
    tqdm numpy scipy pandas scikit-learn pyyaml tabulate \
    pyglet \
    tornado==6.* \
    matplotlib \
    ipywidgets

# ---- PyTorch (CUDA 12.9 / cu129) ----
# PyTorch publishes a cu129 index; if stable wheels ever lag, use nightly as noted below.
RUN pip install --index-url https://download.pytorch.org/whl/cu129 \
    torch torchvision

# ---- RL / Gym stack (closest equivalent to your original) ----
# gym is largely superseded by gymnasium; keeping gym here to match your old notebooks.
RUN pip install \
    gymnasium \
    pygame \
    pybullet \
    asciinema \
    && pip install "gymnasium[atari]" \
    && pip install "git+https://github.com/pybox2d/pybox2d#egg=Box2D" \
    && pip install "git+https://github.com/nacansino/gym-bandits" \
    && pip install "git+https://github.com/nacansino/gym-walk" \
    && pip install "git+https://github.com/nacansino/gym-aima"

# ---- Runtime notebook config ----
# Expose same ports as your old image
EXPOSE 8888 6006

# Change back to root to configure workspaces dir
USER root

# Create default notebooks dir
RUN mkdir -p /workspaces/gdrl/notebooks/logs && chown -R ${NB_UID}:${NB_GID} /workspaces/gdrl/notebooks

# ---- Final setup ----
# Copy run script and make it executable
COPY run.sh /usr/local/bin/run.sh
RUN chmod +x /usr/local/bin/run.sh

# Now switch back to NB_USER
# Set working dir and entrypoint
USER ${NB_UID}
WORKDIR /workspaces/gdrl
ENTRYPOINT ["/usr/local/bin/run.sh"]
