FROM linux:centos9

# maca
RUN --mount=type=bind,source=./driver,target=/mnt/driver /mnt/driver/mxdriver-install.sh -m umd

ENV PATH=/opt/mxdriver/bin:$PATH \
    LIBRARY_PATH=/opt/mxdriver/lib:$LIBRARY_PATH \
    LD_LIBRARY_PATH=/opt/mxdriver/lib:$LD_LIBRARY_PATH \
    C_INCLUDE_PATH=/opt/mxdriver/include/mxsml:$C_INCLUDE_PATH \
    CPLUS_INCLUDE_PATH=/opt/mxdriver/include/mxsml:$CPLUS_INCLUDE_PATH 


RUN --mount=type=bind,source=./sdk,target=/mnt/sdk /mnt/sdk/.layerspec/install.sh

# sdk
RUN --mount=type=bind,source=./sdk,target=/mnt/sdk  \
    /mnt/sdk/mxmaca-sdk-install.sh -f && \
    echo "/opt/maca/lib"             > /etc/ld.so.conf.d/maca.conf && \
    echo "/opt/maca/mxgpu_llvm/lib" >> /etc/ld.so.conf.d/maca.conf && \
    ldconfig

ENV PATH=/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/ompi/bin:/opt/maca/ucx/bin:$PATH \
    MACA_PATH=/opt/maca \
    MACA_CLANG_PATH=/opt/maca/mxgpu_llvm/bin \
    LD_LIBRARY_PATH=/opt/maca/ompi/lib:/opt/maca/ucx/lib:$LD_LIBRARY_PATH

# pytorch
RUN --mount=type=bind,source=./pytorch,target=/mnt/pytorch /mnt/pytorch/.layerspec/install.sh

# ray
ARG CUCC_TARGETS="xcore1000"

LABEL com.metax.driver.version="3.1.0.11"
LABEL com.metax.sdk.version="3.1.0.14"
LABEL com.metax.torch.version="2.4+3.1.0.4"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
RUN mkdir -p /workspace /opt/thirdparty/lib
RUN --mount=type=bind,source=./ray,target=/mnt/ray /mnt/ray/.layerspec/install.sh

ENV CUCC_PATH=/opt/maca/tools/cu-bridge \
    CUDA_PATH=/opt/maca/tools/cu-bridge

RUN ldconfig && \
    chmod -R 777 /workspace


WORKDIR /workspace

# llvm
ARG CUCC_TARGETS="xcore1000"

LABEL com.metax.driver.version="3.1.0.11"
LABEL com.metax.sdk.version="3.1.0.14"
LABEL com.metax.torch.version="2.6+3.1.0.4"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
RUN mkdir -p /workspace /opt/thirdparty/lib
RUN --mount=type=bind,source=./vllm,target=/mnt/vllm /mnt/vllm/.layerspec/install.sh

ENV CUCC_PATH=/opt/maca/tools/cu-bridge \
    CUDA_PATH=/opt/maca/tools/cu-bridge

RUN ldconfig && \
    chmod -R 777 /workspace


WORKDIR /workspace
