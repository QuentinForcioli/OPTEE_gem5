

CP?= cp
RM?= rm
MV?= mv
RMDIR?= rmdir
MKDIR?= mkdir
SCONS?= scons

#GEM5
CROSS_COMPILE?= aarch64-linux-gnu-
CROSS_COMPILE_32?= arm-linux-gnueabihf- 
CROSS_COMPILE_64?= aarch64-linux-gnu-


PWD=$(shell pwd)
GEM5?=build/ARM/gem5.opt
GEM5_PATH?=$(realpath $(PWD)/../../../../)
GEM5_DBG_CL:=$(GEM5_PATH)/$(GEM5)
GEM5_CL=$(GEM5_PATH)/$(GEM5)
#M5_PATH
COMMON_SYSTEM=$(realpath $(PWD)/../../../common/system/arm/)
EXAMPLE_SYSTEM=$(realpath $(PWD)/system/)

JOB_COUNT= 10


all: binaries/bl1.bin binaries/fip.bin binaries/u-boot.bin disks/rootfs.ext2 semihosting/Image 

#****************************************************************
#linux (5.9 => optee 3.13)
#****************************************************************
linux:
	git clone https://github.com/linaro-swg/linux.git -b optee-3.12.0
	cp linux.config linux/.config
linux-defconfig: 
	make ARCH=arm64 CROSS_COMPILE=$(CROSS_COMPILE) -C linux defconfig
linux-config: linux
	make ARCH=arm64 CROSS_COMPILE=$(CROSS_COMPILE) -C linux menuconfig
linux-clean: linux
	make ARCH=arm64 CROSS_COMPILE=$(CROSS_COMPILE) -C linux distclean	
semihosting/Image: linux
	make ARCH=arm64 CROSS_COMPILE=$(CROSS_COMPILE) -C linux -j $(JOB_COUNT)
	mkdir -p semihosting
	cp linux/arch/arm64/boot/Image ./semihosting
#****************************************************************
#u-boot (a)
#****************************************************************
#Delay in seconds before automatically booting -> 0 :
#	Skip the waiting period for auto-booting. Note: you can leave this as 1 
#	and press any key during the waiting period to get into U-Boot shell to do a 
#	manual boot. Access to the shell has been tested, but not manual booting.
#Boot arguments
#	-Remove "debug user_debug=31 loglevel=9"
#	-Append "root=/dev/vda rw norandmaps"
#Environment
#	-Environment is not stored -> Enabled
#	-Environment in flash memory -> Disabled
#		No Flash memory support.
#Networking Support -> Disabled
#	-#No support for the Ethernet controller in the Foundation platform, so we disable network support for avoiding crash on access

u-boot:
	git clone https://github.com/u-boot/u-boot.git -b v2020.07-rc3
	cp u-boot.config u-boot/.config
u-boot_defconfig: u-boot
	make CROSS_COMPILE=$(CROSS_COMPILE) -C u-boot vexpress_aemv8a_semi_config
u-boot_config: u-boot
	make CROSS_COMPILE=$(CROSS_COMPILE) -C u-boot menuconfig
u-boot_clean: u-boot
	make CROSS_COMPILE=$(CROSS_COMPILE) -C u-boot clean
u-boot_build: u-boot_clean
	cp u-boot.config u-boot/.config
	make CROSS_COMPILE=$(CROSS_COMPILE) -C u-boot DEBUG=1 -j $(JOB_COUNT)	
binaries/u-boot.bin: u-boot
	cp u-boot.config u-boot/.config
	make CROSS_COMPILE=$(CROSS_COMPILE) -C u-boot DEBUG=1 -j $(JOB_COUNT)
	mkdir -p binaries
	mv u-boot/u-boot.bin ./binaries
#****************************************************************
#trusted-firmware
#****************************************************************
trusted-firmware-a:
	git clone https://git.trustedfirmware.org/TF-A/trusted-firmware-a.git -b v2.3

TFA_OPTEE_FLAG:=BL32=../binaries/tee-header_v2.bin\
	 BL32_EXTRA1=../binaries/tee-pager_v2.bin \
	 BL32_EXTRA2=../binaries/tee-pageable_v2.bin \
	 ARM_TSP_RAM_LOCATION=tdram \
	 BL32_RAM_LOCATION=tdram \
	 SPD=opteed
#TFA_OPTEE_FLAG:= 

binaries/bl1.bin binaries/fip.bin: trusted-firmware-a binaries/u-boot.bin binaries/tee.bin
	make trusted-firmware-a_clean
	make CROSS_COMPILE=$(CROSS_COMPILE) \
	 LOG_LEVEL=40\
     PLAT=fvp \
	 DEBUG=1 \
	 EL3_EXCEPTION_HANDLING=0\
     BL33=../binaries/u-boot.bin \
	 $(TFA_OPTEE_FLAG) \
     -C trusted-firmware-a all fip
	mkdir -p binaries
	mv trusted-firmware-a/build/fvp/debug/bl1.bin ./binaries
	mv trusted-firmware-a/build/fvp/debug/fip.bin ./binaries	
#	 EL3_EXCEPTION_HANDLING=1  RESET_TO_BL31=1 SDEI_SUPPORT=1 EL3_EXCEPTION_HANDLING=1 \ 
trusted-firmware-a_clean:
	make -C trusted-firmware-a realclean PLAT=fvp
#****************************************************************
#build-root
#****************************************************************
#Target Options:
#	- Target Architecture Variant -> cortex-A57
#Toolchain :
#	- Kernel Headers -> Manually specified Linux version
#	- linux version -> 5.4
#	- Custom kernel headers series -> 5.4.x or later
#Kernel -> Disabled:
#	-We build the Linux Kernel externally.
#Target packages:
#	BusyBox configuration file to use?:
#		-package/busybox/busybox-minimal.config
#	Networking applications:
#		-ifupdown scripts -> Disabled :We do not support networking.
BUILDROOT_OUT=out-br
BUILDROOT_CONF=buildroot.config
double-quote = "#" # This really sets the variable to " and avoids upsetting vim's syntax highlighting
streq = $(and $(findstring $(1),$(2)),$(findstring $(2),$(1)))
y-or-n = $(or $(call streq,y,$(1)),$(call streq,n,$(1)))
append-var_ = echo '$(1)=$(3)'$($(1))'$(3)' >>$(2);
append-var = $(call append-var_,$(1),$(2),$(if $(call y-or-n,$($(1))),,$(double-quote)))
append-br2-vars = $(foreach var,$(filter BR2_%,$(.VARIABLES)),$(call append-var,$(var),$(1)))
BR2_aarch64=y
BR2_PACKAGE_LIBOPENSSL ?=y
BR2_PACKAGE_MMC_UTILS ?=y
BR2_PACKAGE_OPENSSL ?=y
BR2_PACKAGE_OPTEE_OS_EXT=y
BR2_TARGET_ROOTFS_CPIO=y
BR2_TARGET_ROOTFS_CPIO_GZIP=y
BR2_PACKAGE_HOST_E2FSPROGS=y
BR2_PACKAGE_OPTEE_CLIENT_EXT=y
BR2_PACKAGE_OPTEE_CLIENT_EXT_SITE=$(CURDIR)/optee_client
BR2_PACKAGE_OPTEE_OS_EXT_SDK=$(CURDIR)/optee_os/out/arm/export-ta_arm64 
BR2_PACKAGE_OPTEE_OS_EXT_SITE=$(CURDIR)/br-ext/package/optee_os_ext
BR2_PACKAGE_OPTEE_TEST_EXT=y
BR2_PACKAGE_OPTEE_TEST_EXT_CROSS_COMPILE?=$(CROSS_COMPILE)
BR2_PACKAGE_OPTEE_TEST_EXT_SDK=$(CURDIR)/optee_os/out/arm/export-ta_arm64
BR2_PACKAGE_OPTEE_TEST_EXT_SITE=$(CURDIR)/optee_test
BR2_PACKAGE_OPTEE_TEST_EXT_GP_PACKAGE := $(GP_PACKAGE)
BR2_PACKAGE_OPTEE_TEST_EXT_WITH_TLS_TESTS:=y
BR2_PACKAGE_OPTEE_TEST_EXT_WITH_CXX_TESTS:=y

m5: out-br 
	$(SCONS) -C $(GEM5_PATH)/util/m5 build/arm64/out/m5 
	mkdir -p out-br/target/usr/bin
	cp -rf $(GEM5_PATH)/util/m5/build/arm64/out/m5 out-br/target/usr/bin
	mkdir -p out-br/target/etc/init.d/
	cp S99auto_m5_rcS.sh out-br/target/etc/init.d/S99auto_m5_rcS.sh

buildroot:
	git clone https://github.com/buildroot/buildroot.git -b 2020.08
	make ARCH=aarch64 BR2_JLEVEL=$(JOB_COUNT) CROSS_COMPILE=$(CROSS_COMPILE) -C buildroot/ arm_foundationv8_defconfig

out-br: optee_test optee_client buildroot
	@mkdir -p out-br/target
	@rm -f out-br/build/optee_*/.stamp_*
	@rm -f out-br/extra.conf
	@$(call append-br2-vars,out-br/extra.conf)
	@(python3 br-ext/scripts/make_def_config.py \
		--br buildroot --out out-br --br-ext br-ext \
		--top-dir "$(CURDIR)" \
		--br-defconfig br-ext/configs/optee_aarch64 \
		--br-defconfig br-ext/configs/optee_generic \
		--br-defconfig out-br.config \
		--br-defconfig out-br/extra.conf \
		--make-cmd $(MAKE))
buildroot_out-br_config: out-br
	make ARCH=aarch64 BR2_JLEVEL=$(JOB_COUNT) CROSS_COMPILE=$(CROSS_COMPILE) -C out-br/ menuconfig 
out-br/images/rootfs.ext2: out-br m5
	$(MAKE) -C out-br  ARCH=aarch64 BR2_JLEVEL=$(JOB_COUNT) CROSS_COMPILE=$(CROSS_COMPILE)

buildroot_clean:
	rm -rf out-br

disks/rootfs.ext2: out-br/images/rootfs.ext2
	mkdir -p disks
	 mv out-br/images/rootfs.ext2 ./disks/
	

#***************************************************
#OP TEE
#***************************************************
optee_os:
	git clone https://github.com/OP-TEE/optee_os.git -b 3.12.0
optee_client:
	git clone https://github.com/OP-TEE/optee_client -b 3.12.0
optee_test:
	git clone https://github.com/OP-TEE/optee_test -b 3.12.0
optee_os/out/arm/export-ta_arm64 optee_os/out/arm/export-ta_arm32  binaries/tee.bin: optee_os
	make -C optee_os clean
	make -C optee_os \
    CFG_ARM64_core=y \
    CFG_TEE_BENCHMARK=y \
    CFG_TEE_CORE_LOG_LEVEL=4 \
	CFG_ARM_GICV3=y \
    CROSS_COMPILE=$(CROSS_COMPILE_64) \
    CROSS_COMPILE_core=$(CROSS_COMPILE_64) \
    CROSS_COMPILE_ta_arm32=$(CROSS_COMPILE_32) \
    CROSS_COMPILE_ta_arm64=$(CROSS_COMPILE_64) \
	CFG_WITH_VFP=n \
	CFG_TA_FLOAT_SUPPORT=n \
    DEBUG=1 \
    CFG_WITH_ARM_TRUSTED_FW=y \
    O=out/arm PLATFORM=vexpress-fvp
	mkdir -p binaries
	mv optee_os/out/arm/core/tee.bin ./binaries
	mv optee_os/out/arm/core/tee-header_v2.bin ./binaries
	mv optee_os/out/arm/core/tee-pager_v2.bin ./binaries
	mv optee_os/out/arm/core/tee-pageable_v2.bin ./binaries
optee_client/out/export: optee_client 
	mkdir -p optee_client/build
	mkdir -p optee_client/install
	cmake -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc -DCMAKE_INSTALL_PREFIX=optee_client/install/ -Hoptee_client -Boptee_client/build
	make -C optee_client CROSS_COMPILE=aarch64-linux-gnu-
	make -C optee_client CROSS_COMPILE=aarch64-linux-gnu- install
#sudo apt-get install android-tools-adb android-tools-fastboot autoconf \
#        automake bc bison build-essential ccache cscope curl device-tree-compiler \
#        expect flex ftp-upload gdisk  libattr1-dev libcap-dev \
#        libfdt-dev libftdi-dev libglib2.0-dev libhidapi-dev libncurses5-dev \
#        libpixman-1-dev libssl-dev libtool make \
#        mtools netcat python-crypto python3-crypto python-pyelftools \
#        python3-pycryptodome python3-pyelftools python3-serial \
#        rsync unzip uuid-dev xdg-utils xterm xz-utils zlib1g-dev\
#this last one are not available under ubuntu 20.4 but they don't seem to be needed for the build to work:
#sudo apt-get install iasl python-serial 
export: optee_client/out/export optee_os/out/arm/export-ta_arm64 optee_os/out/arm/export-ta_arm32
	mkdir export
	cp -rf optee_os/out/arm/export-ta_arm32 export/
	cp -rf optee_os/out/arm/export-ta_arm64 export/
	mkdir export/client_arm64
	cp -rf optee_client/out/export/usr/lib export/client_arm64/lib
	cp -rf optee_client/out/export/usr/include export/client_arm64/include
clean_sources: 
	$(RM) -rf linux
	$(RM) -rf buildroot
	$(RM) -rf trusted-firmware-a
	$(RM) -rf u-boot
	$(RM) -rf optee_client
	$(RM) -rf optee_os
	$(RM) -rf optee_test
clean_build: 
	$(RM) -rf binaries
	$(RM) -rf disks
	$(RM) -rf semihosting
	$(RM) -rf export
	$(RM) -rf out-br
clean_all:clean_build clean_sources
clean:clean_build 
	make -C linux clean
	make -C buildroot clean
	make -C trusted-firmware-a clean
	make -C u-boot clean
	make -C optee_client clean
	make -C optee_os clean
	make -C optee_test clean
