ARCH := SM120
SRC := bf16_sm120_mha_causal.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python test.py
CONFIG := pytorch

include ../../common.mk
