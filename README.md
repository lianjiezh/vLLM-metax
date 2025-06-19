# vLLM-MetaX

The vLLM-MetaX backend plugin for vLLM.

## Installation

Currently we only support build from source:

```bash
# install vllm
pip install vllm==0.8.5 --no-deps

# install vllm-metax
git clone  --depth 1 --branch v0.8.5 [vllm-metax-repo]
cd vllm-metax

source env.sh
python setup.py bdist_wheel
pip install dist/vllm_metax_plugin-0.8.5*.whl
```

> Note: plugin would copy the `.so` files to the vllm_dist_path, which is the `vllm` under `pip show vllm | grep Location` by default.
>
> If you :
>
> - ***skipped the build step*** and installed the binary distribution `.whl` from somewhere(e.g. pypi) else
>
> - Or ***reinstalled*** the official vllm
>
> You need **manually** executing the following command to initialize the plugin :

```bash
$ vllm_metax_init
```