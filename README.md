# vLLM-MetaX

The MXMACA backend plugin for vLLM.

## Install plugin

Currently we only support building in docker.
> if build in host, you need to manually install all the dependencies and package requirements first.

### install vllm
**clone repository and install from source**:
```bash
# clone vllm
git clone  --depth 1 --branch main https://github.com/vllm-project/vllm && cd vllm

# install build requirements
python use_existing_torch.py
pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install -v . --no-build-isolation
```

### install vllm-metax

**setup env variables**:

```
# setup MACA path
export MACA_PATH="/opt/maca"

# cu-bridge
export CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
export CUDA_PATH=/root/cu-bridge/CUDA_DIR
export CUCC_CMAKE_ENTRY=2

# update PATH
export PATH=${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}

export VLLM_INSTALL_PUNICA_KERNELS=1
```

**clone repository**:
```bash
# clone vllm-metax
git clone  --depth 1 --branch [branch-name] [vllm-metax-repo-url] && cd vllm-metax
```
There are two ways to build the plugin:

> build on *released* docker image
> we need to add `-no-build-isolation` flag (or an equivalent one) during package building. 
> Since all the requirements are already pre-installed in released docker image.

- if you want to build the binary distribution:

```bash
# install requirements for building
python use_existing_metax.py
pip install -r requirements/build.txt
# build wheels
python -m build -w -n
# install wheels
pip install dist/*.whl
```

- Or, install directly:

```bash
# install requirements for building
python use_existing_metax.py
pip install -r requirements/build.txt
# since we use our local pytorch, add the --no-build-isolation flag 
# to avoid the conflict with the official pytorch
pip install . -v --no-build-isolation
```
