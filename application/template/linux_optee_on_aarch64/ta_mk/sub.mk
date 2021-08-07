global-incdirs-y += $(shell realpath --relative-to $(shell pwd) $(TA_INCLUDE_DIRS))
srcs-y += $(shell realpath --relative-to $(shell pwd) $(TA_SRC_LIST))
base-prefix :=../obj/ta/ta/
# To remove a certain compiler flag, add a line like this
#cflags-template_ta.c-y += -Wno-strict-prototypes
