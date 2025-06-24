FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 1) System deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
       blender \
       python3-pip \
       python3-dev \
       libgl1-mesa-glx \
       libglib2.0-0 \
       assimp-utils \
       libassimp-dev \
       git \
       wget \
       cmake \
       build-essential \
       ninja-build \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install "numpy<2"
RUN pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip3 install fvcore iopath
RUN git clone https://github.com/NVIDIA/cub.git /opt/cub
ENV CUB_HOME=/opt/cub
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.5"

RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# 2) Python deps
RUN pip3 install \
       torch torchvision \
       trimesh \
       pillow \
       pyassimp \
       usd-core \
       open3d \
       pygltflib \
       imageio \
       lpips \
       piq

RUN wget https://damassets.autodesk.net/content/dam/autodesk/www/files/fbx202037_fbxsdk_gcc_linux.tar.gz -O /tmp/FBXSDK_linux.tgz
RUN mkdir -p /opt/fbxsdk && \
    tar -xvzf /tmp/FBXSDK_linux.tgz -C /opt/fbxsdk && \
    yes yes | /opt/fbxsdk/fbx202037_fbxsdk_linux /opt/fbxsdk && \
    rm /tmp/FBXSDK_linux.tgz

RUN wget https://damassets.autodesk.net/content/dam/autodesk/www/files/fbx202037_fbxpythonsdk_linux.tar.gz -O /tmp/FBXPythonSDK_linux.tgz
RUN tar -xvzf /tmp/FBXPythonSDK_linux.tgz -C /tmp/ && \
    yes yes | /tmp/fbx202037_fbxpythonsdk_linux /opt/fbxsdk && \
    rm /tmp/FBXPythonSDK_linux.tgz

RUN pip3 install /opt/fbxsdk/fbx-2020.3.7-cp310-cp310-manylinux1_x86_64.whl

ENV FBX_SDK_ROOT=/opt/fbxsdk
ENV LD_LIBRARY_PATH=$FBX_SDK_ROOT/lib/gcc4_x64:$LD_LIBRARY_PATH
ENV PYTHONPATH=$FBX_SDK_ROOT/lib/Python37_x64:$PYTHONPATH

WORKDIR /app
COPY . /app
COPY static /app/static
