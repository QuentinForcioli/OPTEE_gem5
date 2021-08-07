ifeq ($(GEM5_PATH),)
$(info please define GEM5_PATH)
$(info current value GEM5_PATH=$(GEM5_PATH))
else
optee_demo:
	make -C application/template/linux_optee_on_aarch64 GEM5_PATH=$(realpath $(GEM5_PATH))  gem5_q
endif
