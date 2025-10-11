FROM registry.access.redhat.com/ubi9/ubi:9.6

# 1. Installing Metax-Driver
RUN echo "[metax-centos]" > /etc/yum.repos.d/metax-driver-centos.repo && \
    echo "name=Maca Driver Yum Repository" >> /etc/yum.repos.d/metax-driver-centos.repo && \
    echo "baseurl=https://repos.metax-tech.com/r/metax-driver-centos-$(uname -m)/" >> /etc/yum.repos.d/metax-driver-centos.repo && \
    echo "enabled=1" >> /etc/yum.repos.d/metax-driver-centos.repo && \
    echo "gpgcheck=0" >> /etc/yum.repos.d/metax-driver-centos.repo

# would install the newest 3.1.0.x release
# Metax-Driver mainly contains vbios and kmd file, which are not needed in a container.
# Here we want to get the mx-smi management tool. 
# kernel version mismatch errors are ignored

RUN yum makecache && \
    yum install -y metax-driver mxgvm && \
    yum clean all


# 2. Installing MACA SDK
RUN echo "[maca-sdk]" > /etc/yum.repos.d/maca-sdk-rpm.repo && \
    echo "name=Maca Sdk Yum Repository" >> /etc/yum.repos.d/maca-sdk-rpm.repo && \
    echo "baseurl=https://repos.metax-tech.com/r/maca-sdk-rpm-$(uname -m)/" >> /etc/yum.repos.d/maca-sdk-rpm.repo && \
    echo "enabled=1" >> /etc/yum.repos.d/maca-sdk-rpm.repo && \
    echo "gpgcheck=0" >> /etc/yum.repos.d/maca-sdk-rpm.repo

RUN yum makecache && \
    yum install -y maca_sdk && \
    yum clean all


# 3. Install torch

## 3.1 Setup a python environment using uv

# Our internal build pipeline uses conda. Here we use uv instead. 
# This is NOT fully tested!
RUN curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv_install.sh && \
    sh /tmp/uv_install.sh && \
    source $HOME/.local/bin/env && \
    uv venv /opt/venv --python 3.10 && \
    rm /tmp/uv_install.sh


## 3.2 Install dependent yum packages

# The following packages need subscription, so we just SKIP them. 
# NOT fully tested, but should not cause big problems.
# `lapack-devel librdmacm-utils libibverbs-utils`

RUN yum makecache && yum install -y \
    wget zip unzip tar tzdata vim git \
    openblas-devel \
    make cmake patch ninja-build gcc gcc-c++ \
    procps-ng libxml2 libXau \
    libibverbs librdmacm libibumad openssh-server \
    && yum clean all


## 3.3 Install cu-bridge

RUN cd /tmp/ && \
    export MACA_PATH=/opt/maca && \
    curl -o 3.1.0.zip -LsSf https://gitee.com/metax-maca/cu-bridge/repository/archive/3.1.0.zip && \
    unzip 3.1.0.zip && \
    mv cu-bridge-3.1.0 cu-bridge && \
    chmod 755 cu-bridge -Rf && \
    cd cu-bridge && \
    mkdir build && cd ./build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/maca/tools/cu-bridge ../ && \
    make && make install


## 3.4 Some import environment settings

ENV MACA_PATH=/opt/maca \
    MACA_CLANG_PATH=/opt/maca/mxgpu_llvm/bin \
    CUCC_PATH=/opt/maca/tools/cu-bridge \
    CUDA_PATH=/opt/maca/tools/cu-bridge \
    PATH=/opt/mxdriver/bin:/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/tools/cu-bridge/tools:/opt/maca/tools/cu-bridge/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/mxdriver/lib:/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:${LD_LIBRARY_PATH}


## 3.5 Install torch using uv 

RUN echo "apex==0.1+metax3.1.0.4" > /tmp/requirements.txt && \
    echo "causal_conv1d==1.5.0.post8+metax3.1.0.4torch2.6" >> /tmp/requirements.txt && \
    echo "dropout_layer_norm==0.1+metax3.1.0.4torch2.6" >> /tmp/requirements.txt && \
    echo "flash_attn==2.6.3+metax3.1.0.4torch2.6" >> /tmp/requirements.txt && \
    echo "flash_linear_attention==0.1+metax3.1.0.4torch2.6" >> /tmp/requirements.txt && \
    echo "flash_mla==1.0.1+metax3.1.0.4torch2.6" >> /tmp/requirements.txt && \
    echo "flashinfer==0.2.2.post1+metax3.1.0.4torch2.6" >> /tmp/requirements.txt && \
    echo "fused_dense_lib==2.6.3+metax3.1.0.4torch2.6" >> /tmp/requirements.txt && \
    echo "mamba_ssm==2.2.4+metax3.1.0.4torch2.6" >> /tmp/requirements.txt && \
    echo "mctlassEx==0.1.1+metax3.1.0.4torch2.6" >> /tmp/requirements.txt && \
    echo "rotary_emb==0.1+metax3.1.0.4torch2.6" >> /tmp/requirements.txt && \
    echo "sageattention==2.0.1+metax3.1.0.4torch2.6" >> /tmp/requirements.txt && \
    echo "spconv==2.1.0+metax3.1.0.4torch2.6" >> /tmp/requirements.txt && \
    echo "torch==2.6.0+metax3.1.0.4" >> /tmp/requirements.txt && \
    echo "torchaudio==2.4.1+metax3.1.0.4" >> /tmp/requirements.txt && \
    echo "torchvision==0.15.1+metax3.1.0.4" >> /tmp/requirements.txt && \
    echo "triton==3.0.0+metax3.1.0.4" >> /tmp/requirements.txt && \
    echo "xentropy_cuda_lib==0.1+metax3.1.0.4torch2.6" >> /tmp/requirements.txt && \
    echo "xformers==0.0.22+metax3.1.0.4torch2.6" >> /tmp/requirements.txt && \
    source $HOME/.local/bin/env && \
    source /opt/venv/bin/activate && \
    uv pip install datasets==4.1.1 && \
    uv pip install -r /tmp/requirements.txt -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com && \
    uv pip install numpy==1.26.4


WORKDIR /workspace


# 4. Build and install vllm

# Reference: [vLLM-metax](https://github.com/MetaX-MACA/vLLM-metax)

# build vllm on maca needs cuda 11.6
RUN curl -o cuda_11.6.0_510.39.01_linux.run -LsSf https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run && \
    sh cuda_11.6.0_510.39.01_linux.run --silent --toolkit && \
    rm cuda_11.6.0_510.39.01_linux.run

# install vllm (or build from source)
RUN source $HOME/.local/bin/env && \
    source /opt/venv/bin/activate && \
    uv pip install /opt/maca/share/mxsml/pymxsml-*.whl && \
    git clone --depth 1 --branch main https://github.com/vllm-project/vllm && \
    cd vllm && \
    python use_existing_torch.py && \
    uv pip install -r requirements/build.txt && \
    VLLM_TARGET_DEVICE=empty uv pip install -v . --no-build-isolation && \
    cd ..

RUN git clone --depth 1 --branch v0.10.2 https://github.com/MetaX-MACA/vLLM-metax.git && \
    cd vLLM-metax && \
    export CUDA_PATH="/usr/local/cuda" && \
    export PATH=/usr/local/cuda/bin::${PATH} && \
    export VLLM_INSTALL_PUNICA_KERNELS=1 && \
    source $HOME/.local/bin/env && \
    source /opt/venv/bin/activate && \
    uv pip install -r requirements/build.txt && \
    python setup.py bdist_wheel && \
    uv pip install dist/*.whl


## 5. Install ray

RUN source $HOME/.local/bin/env && \
    source /opt/venv/bin/activate && \
    uv pip install click==8.2.1 ray==2.46.0

COPY ray-patch.zip /tmp/

RUN cd /tmp && unzip ray-patch.zip && \
    mv ray-patch /workspace && \
    cd /workspace/ray-patch/ray_patch && \
    source $HOME/.local/bin/env && \
    source /opt/venv/bin/activate && \
    python apply_ray_patch.py mx_ray_2.46.batch && \
    if [ -f "/opt/conda/bin/ray" ]; then ln -sf /opt/conda/bin/ray /bin/ray; fi


RUN echo 'source /root/.local/bin/env && source /opt/venv/bin/activate' >> /root/.bashrc
