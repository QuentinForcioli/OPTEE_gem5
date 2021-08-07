#CROSS_COMPILE_DIR = /usr/local/bin
#*******************************************************
#compatibility mode for windows and linux
#*******************************************************
ARCH          ?= aarch64
CROSS_COMPILE ?= aarch64-linux-gnu-
SUFFIX:=
test:= $(shell $(CROSS_COMPILE)gcc.exe --version 2>/dev/null)
ifdef test
test:= $(shell $(CROSS_COMPILE)gcc --version 2>/dev/null)
$(info Windows toolchain found)
ifdef test  
$(info linux toolchain also found : choosing it)
else
SUFFIX:=.exe
endif
else
$(info Windows toolchain not found)
SUFFIX:=
endif
#****************************************
#runable
#****************************************
AS          = $(CROSS_COMPILE)as$(SUFFIX)
LD          = $(CROSS_COMPILE)ld$(SUFFIX)
CC          = $(CROSS_COMPILE)gcc$(SUFFIX)
CXX         = $(CROSS_COMPILE)g++$(SUFFIX)
CPP         = $(CROSS_COMPILE)cpp$(SUFFIX)
AR          = $(CROSS_COMPILE)ar$(SUFFIX)
NM          = $(CROSS_COMPILE)nm$(SUFFIX)
STRIP       = $(CROSS_COMPILE)strip$(SUFFIX)
OBJCOPY     = $(CROSS_COMPILE)objcopy$(SUFFIX)
OBJDUMP     = $(CROSS_COMPILE)objdump$(SUFFIX)
RANLIB      = $(CROSS_COMPILE)ranlib$(SUFFIX)
GDB 		= $(CROSS_COMPILE)gdb$(SUFFIX)
GCC_LIB_DIR		 = $(CROSS_COMPILE:-=)/libc 

CP= cp
RM = rm
MV= mv
RMDIR= rmdir
MKDIR= mkdir
PWD=$(shell pwd)

#*************************************************************
GEM5_PYTHONPATH = $(realpath $(GEM5_PATH)/configs/)
GEM5_PATH?=$(PWD)/../../../../


GEM5 = build/ARM/gem5.opt
GEM5_DBG_CL=$(GEM5_PATH)/build/ARM/gem5.debug
GEM5_CL=$(GEM5_PATH)/$(GEM5)
#*************************************************************

DEFAULT_PYTHONPATH=$(realpath $(PWD)/../../../common/configs/):$(GEM5_PYTHONPATH)
COMPAT_PYTHONPATH=$(realpath $(GEM5_PATH)/configs/example/):$(realpath $(GEM5_PATH)/configs/example/arm/)
export PYTHONPATH:=$(PYTHONPATH):$(PWD)/:$(PWD)/config/:$(DEFAULT_PYTHONPATH):$(COMPAT_PYTHONPATH)

COMMON_SYSTEM=$(realpath $(PWD)/../../../common/system/arm/)
EXAMPLE_SYSTEM=$(realpath $(PWD)/system/)

COMMON_DEFINE = 
ARM_BASE_DEFINE =			


#****************************************************************
#OPTEE-TF-A
#****************************************************************
TFA_DIR=../../lib/optee_tf-a/
export M5_PATH = $(TFA_DIR):$(COMMON_SYSTEM):$(EXAMPLE_SYSTEM)
TEEC_EXPORT=$(TFA_DIR)//out-br/staging/usr
TA_DEV_KIT_DIR=$(TFA_DIR)/optee_os/out/arm/export-ta_arm64
TFA_PRODUCTS = $(TFA_DIR)/binaries/fip.bin  $(TFA_DIR)/disks/rootfs.ext2 $(TFA_DIR)/semihosting/Image \
               $(TEEC_EXPORT)/lib/libteec.so $(TA_DEV_KIT_DIR)/mk/ta_dev_kit.mk
#****************************************************************
#targets
#****************************************************************
TA_BINARY=8aaaf200-2450-11e4-abe2-0002a5d5c51b
TARGET_HOST=bin/main
TARGET_TA=bin/$(TA_BINARY).ta
TARGET_DISK=mounted_disk.ext2
ORIGINAL_TEMPLATE_DIR=../../template/linux_optee_on_aarch64/
#****************************************************************
#LIBS & DISKS
#****************************************************************
LIBS = teec #no libs
ROOT_DISK = $(TFA_DIR)/disks/rootfs.ext2
#****************************************************************
#src dir & object (HOST)
#****************************************************************
HOST_LIB_DIR:=lib/ $(TEEC_EXPORT)/ 
HOST_LIB_LIST=$(LIBS) 
HOST_INCLUDE_DIR=inc include/ host $(addsuffix /include,$(HOST_LIB_DIR)) $(addsuffix /inc,$(HOST_LIB_DIR)) 
HOST_SRC_DIR=src/host/
HOST_SRC_LIST_C=$(wildcard src/host/*.c) $(wildcard src/host/*/*.c) $(wildcard src/host/*/*/*.c) $(wildcard src/host/*/*/*/*.c)
HOST_OBJ_LIST=$(patsubst %.c,%.o,$(patsubst src/%,obj/%,$(HOST_SRC_LIST_C)))
HOST_OBJ_LD_LIST= $(HOST_OBJ_LIST)
HOST_DEP=$(wildcard $(addsuffix /*,$(HOST_INCLUDE_DIR)))
COMMON_DEP =Makefile 
#******************************************************************************************************************
#src dir & object (TA)
#******************************************************************************************************************
TA_SRC_LIST=$(wildcard src/ta/*.c) $(wildcard src/ta/*/*.c) $(wildcard src/ta/*/*/*.c) $(wildcard src/ta/*/*/*/*.c)
TA_INCLUDE_DIRS=include/ta include/
#******************************************************************************************************************
#folder structure
#*********************************************************************
$(shell $(MKDIR) -p bin)
#*********************************************************************
#images directory
#*********************************************************************
TARGET_DISKS_DIR=mnt
TARGET_DISKS_MNT=$(TARGET_DISKS_DIR)_mnt
#*********************************************************************
#User configuration 
#*********************************************************************
ifndef CPU_COUNT
CPU_COUNT=1
endif

#***************************************************
#tunable flags
#***************************************************
OPTI=-g -O2
WARNING = -Wall
ARCH= -mabi=lp64 -march=armv8.1-a
#*****************************************************
#GEM5
#*****************************************************
CONFIG_FILE=optee_bm.py
GEM5_OPTIONS= -v
GEM5_DEBUG_FLAGS= --debug-flags=Exec,PMUVerbose,Faults,Arm,Cache
GEM5_RELEASE_FLAGS=--debug-flags=Exec
GEM5_QUIET_FLAGS=
GEM5_GDB_FLAGS=--debug-flags=Faults,ITS,Interrupt,GIC,Semihosting
CONFIG_CPUS_OPTIONS= --num-cores 1 --cpu-freq 1GHz --cpu atomic
CONFIG_MEM_OPTIONS= --mem-type DDR3_1600_8x8 --mem-size 4GB
CONFIG_OPTIONS= --machine-type=VExpress_GEM5_Foundation \
				--workload ArmTrustedFirmware\
				$(CONFIG_CPUS_OPTIONS) $(CONFIG_MEM_OPTIONS)\
				--semi-enable --semi-path system/semihosting/ \
				--disk-image=$(ROOT_DISK) --secondary-disk=$(TARGET_DISK)

CONFIG_OPTIONS_GDB= --param 'system.workload.wait_for_remote_gdb = True'  --checkpoint
ifeq (,$(wildcard m5out/cpt.*))
	CONFIG_RESTORE_CHECKPOINT=
else
	CONFIG_RESTORE_CHECKPOINT=--restore=$(wildcard m5out/cpt.*)
endif
CONFIG_AUTO_OPTIONS= --readfile=bash.rcS $(CONFIG_RESTORE_CHECKPOINT)

CONFIG_FILE_PATH=$(CONFIG_FILE)
CHECKPOINT_DIR=$(wildcard m5out/cpt.*)
#***************************************************************************
#GDB
#***************************************************************************
BOOT_SYMBOL_FILES= $(TFA_DIR)/optee_os/out/arm/core/tee.elf $(TFA_DIR)/trusted-firmware-a/build/fvp/debug/bl2/bl2.elf\
 $(TFA_DIR)/trusted-firmware-a/build/fvp/debug/bl1/bl1.elf $(TFA_DIR)/trusted-firmware-a/build/fvp/debug/bl31/bl31.elf\
  $(TFA_DIR)/trusted-firmware-a/build/fvp/debug/bl2u/bl2u.elf $(TFA_DIR)/linux/vmlinux $(TFA_DIR)/u-boot/u-boot
   
BOOT_SYMBOL_DIR=$(TFA_DIR)/optee_os $(TFA_DIR)/trusted-firmware-a

GDB_FLAGS= -ex 'set confirm off'  $(addprefix -ex 'dir ,$(addsuffix ',$(BOOT_SYMBOL_DIR)))\
	$(subst >, ,$(addprefix -ex 'add-symbol-file ,$(addsuffix ',$(BOOT_SYMBOL_FILES)))) \
	-ex 'set remote Z-packet on'
	
#***************************************************************************
#AUTO FLAGS
#***************************************************************************
DEFINE= $(KERNEL_DEFINE) $(COMMON_DEFINE) $(ARM_BASE_DEFINE)
INCLUDES=$(addprefix -I,$(HOST_INCLUDE_DIR)) 
CFLAGS       = $(OPTI) $(INCLUDES) $(WARNING) -std=c11 $(ARCH)
CXXFLAGS     = $(OPTI) $(INCLUDES) $(WARNING) -fno-exceptions $(ARCH)
ASFLAGS      = -EL $(INCLUDES) $(WARNING)
INCLUDE_LIB_DIR = $(addprefix -L,$(HOST_LIB_DIR)) $(addprefix -L,$(GCC_LIB_DIR)) 	
INCLUDE_LIB =    $(addprefix -l,$(notdir $(LIBS)))
LD_FLAGS =  $(COMMON_DEFINE) -lc 
#**************************************************************************
export TA_SRC_LIST:=$(realpath $(TA_SRC_LIST)) 
export TA_INCLUDE_DIRS:=$(realpath $(TA_INCLUDE_DIRS))
#**************************************************************************
#***************************************************************************				 
#rules	
#***************************************************************************
.PHONY: lib lib/%.a phony_rule clean update_makefile

default: $(TARGET_DISK)
system_semihosting: system/semihosting/devtree.dtb system/semihosting/Image
required_to_run: system_semihosting $(TFA_PRODUCTS) $(GEM5_CL)

empty:
ta: $(TARGET_TA)
host: $(TARGET_HOST)
all: $(OBJ_LIST) $(TARGET_HOST) $(TARGET_DISK) $(TARGET_TA)
arm_tfa_optee: $(TFA_PRODUCTS)
#rool for updating makefile
update_makefile:
ifeq ($(PWD),$(realpath $(PWD)/$(ORIGINAL_TEMPLATE_DIR)))
		$(info original directory)
else
		$(info updating)
		$(MV) -f $(PWD)/Makefile $(PWD)/Makefile.old
		$(CP) -f $(PWD)/$(ORIGINAL_TEMPLATE_DIR)/Makefile $(PWD)/Makefile
endif
#***************************************************************************				 
#terminal
#***************************************************************************
terminal:
	telnet localhost 3456
m5term:
	$(GEM5_PATH)/util/term/m5term 3456
m5term_tee:
	$(GEM5_PATH)/util/term/m5term 3457
#***************************************************************************				 
#gem5
#***************************************************************************
gem5_boot:  required_to_run $(TARGET_DISK)
	$(GEM5_CL) $(GEM5_RELEASE_FLAGS) $(GEM5_OPTIONS) $(CONFIG_FILE_PATH) $(CONFIG_OPTIONS) 
gem5_boot_q:  required_to_run $(TARGET_DISK)
	$(GEM5_CL) $(GEM5_QUIET_FLAGS) $(GEM5_OPTIONS) $(CONFIG_FILE_PATH) $(CONFIG_OPTIONS) 
gdb_gem5_boot_q:
	gdb  -ex 'dir $(GEM5_PATH)' --args $(GEM5_CL) $(GEM5_QUIET_FLAGS) $(GEM5_OPTIONS) $(CONFIG_FILE_PATH) $(CONFIG_OPTIONS) 
gem5:  $(TARGET_DISK) required_to_run
	$(GEM5_CL) $(GEM5_RELEASE_FLAGS) $(GEM5_OPTIONS) $(CONFIG_FILE_PATH) $(CONFIG_OPTIONS) $(CONFIG_AUTO_OPTIONS)
gem5_q:  $(TARGET_DISK) required_to_run
	$(GEM5_CL) $(GEM5_QUIET_FLAGS) $(GEM5_OPTIONS) $(CONFIG_FILE_PATH) $(CONFIG_OPTIONS) $(CONFIG_AUTO_OPTIONS)
gem5_d:  $(TARGET_DISK) required_to_run
	$(GEM5_CL) $(GEM5_DEBUG_FLAGS) $(GEM5_OPTIONS) $(CONFIG_FILE_PATH) $(CONFIG_OPTIONS) $(CONFIG_AUTO_OPTIONS)
gem5_dd:  $(TARGET_DISK)  required_to_run
	$(GEM5_DBG_CL) $(GEM5_DEBUG_FLAGS) $(GEM5_OPTIONS) $(CONFIG_FILE_PATH) $(CONFIG_OPTIONS) $(CONFIG_AUTO_OPTIONS)
gem5_i: $(TARGET_DISK) required_to_run
	$(GEM5_CL) -i $(CONFIG_FILE_PATH)  $(CONFIG_OPTIONS)
gem5_h:
	$(GEM5_CL) $(CONFIG_FILE_PATH) --help
gem5_cl: required_to_run
	$(GEM5_CL) $(CONFIG_FILE_PATH) $(CL)
#***************************************************************************				 
#gem5 gdb debug
#***************************************************************************
gem5_boot_gdb:  system_semihosting $(TARGET_DISK)
	$(GEM5_CL) $(GEM5_GDB_FLAGS) $(GEM5_OPTIONS) $(CONFIG_FILE_PATH) $(CONFIG_OPTIONS) $(CONFIG_OPTIONS_GDB)  $(CONFIG_RESTORE_CHECKPOINT)
gdb_gem5_boot_gdb:  system_semihosting $(TARGET_DISK)
	gdb  -ex 'dir $(GEM5_PATH)' --args $(GEM5_DBG_CL)  $(GEM5_GDB_FLAGS) $(GEM5_OPTIONS) $(CONFIG_FILE_PATH) $(CONFIG_OPTIONS) $(CONFIG_OPTIONS_GDB)
tee_gdb: 
	$(GDB) $(GDB_FLAGS) $(TFA_DIR)/optee_os/out/arm/core/tee.elf -ex 'target remote localhost:7000'
tee_gdb_cpu1: 
	$(GDB) $(GDB_FLAGS) $(TFA_DIR)/optee_os/out/arm/core/tee.elf -ex 'target remote localhost:7001'
#****************************************************************************
#building target excecutable (the mounted-copy is automatically rename main)
#****************************************************************************
$(TARGET_HOST): $(HOST_OBJ_LD_LIST) $(COMMON_DEP) $(LIB_LIST) lib/libteec.so  
	$(CC) $(LD_FLAGS) -o $@  $(HOST_OBJ_LD_LIST) $(INCLUDE_LIB_DIR) $(INCLUDE_LIB) 
	$(MKDIR) -p $(TARGET_DISKS_DIR)
	$(CP) $(TARGET_HOST) $(TARGET_DISKS_DIR)/main

$(TARGET_TA): $(TFA_PRODUCTS)
	make -C ta_mk  CROSS_COMPILE=$(CROSS_COMPILE) O=$(realpath bin/) \
	TA_BINARY=$(TA_BINARY) TA_DEV_KIT_DIR=$(realpath $(TA_DEV_KIT_DIR)) LDFLAGS="" 
	$(MKDIR) -p $(TARGET_DISKS_DIR) 
	$(MKDIR) -p $(TARGET_DISKS_DIR)/ta
	$(CP) $(TARGET_TA) $(TARGET_DISKS_DIR)/ta/


$(TARGET_DISK): $(wildcard $(TARGET_DISKS_DIR)/*) Makefile $(TARGET_TA) $(TARGET_HOST) $(TFA_PRODUCTS)
	rm -f $(TARGET_DISK)
	mke2fs -b 1024 -L "OPTEE-archisec" -d $(TARGET_DISKS_DIR) -r 1 -m 5 -t ext2 -N 0 -O ^64bit $(TARGET_DISK) 32M
	
$(GEM5_CL):
	scons -C $(GEM5_PATH) $(GEM5) -j10
$(TFA_PRODUCTS):
	make -C $(TFA_DIR)

lib: $(LIB_LIST) $(COMMON_DEP)
lib/libteec.so: $(TFA_PRODUCTS)
	rm -f lib/libteec.so
	$(MKDIR) -p lib
	ln -s $(realpath $(TEEC_EXPORT)/lib/libteec.so) lib/libteec.so
#**********************************************
#SEMIHOSTING
#**********************************************
system/semihosting/devtree.dtb: Makefile  $(TFA_PRODUCTS)
	$(MKDIR) -p system/semihosting
	$(GEM5_CL) $(CONFIG_FILE_PATH) $(CONFIG_OPTIONS) --dtb-gen 
	mv m5out/system.dtb system/semihosting/devtree.dtb
system/semihosting/Image: $(TFA_PRODUCTS)
	$(MKDIR) -p system/semihosting
	rm -f system/semihosting/Image 
	ln -s ../../$(TFA_DIR)/semihosting/Image system/semihosting/Image 

#**********************************************
#object files
#**********************************************
lib/%.a: phony_rule
	$(MAKE) -C ../../$(dir $@)
	
obj/%.o: src/%.c $(COMMON_DEP)
	$(MKDIR) -p $(dir $@)
	$(CC) $(CFLAGS) $(DEFINE) -o $@ -c $< 
#**********************************************
#cleaning
#**********************************************

clean_obj/%.o: 
	$(RM) -f $(subst clean_, ,$@)
clean_bin/%: 
	$(RM) -f $(subst clean_, ,$@)
clean_m5out:
	$(RM) -fr m5out
clean:
	$(RM) -fr bin
	$(RM) -fr obj
	$(RM) -fr m5out
	$(RM) -f $(TARGET_DISK)
	$(RM) -fr mnt

