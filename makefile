# Config options
EXE    = cuda_raytracing
SRC    = src
BIN    = bin
CC     = nvcc
SHELL  = /bin/bash
CFLAGS = --std=c++17

# Shell colours
BLACK =\u001b[30;1m
RED   =\u001b[31;1m
GREEN =\u001b[32;1m
YELLOW=\u001b[33;1m
BLUE  =\u001b[34;1m
PINK  =\u001b[35;1m
CYAN  =\u001b[36;1m
WHITE =\u001b[37;1m
RESET =\u001b[0m

# Detect platform

ifeq ($(OS),Windows_NT)
    PLATFORM := Windows
else
    PLATFORM := $(shell uname)
endif

export PATH := $(VC_PATH):$(CUDA_PATH)/$(BIN):$(PATH)

# List all of the cu files
CU_SOURCES := $(shell find $(SRC) -name '*.cu')

# List all of the cpp sources
CPP_SOURCES := $(shell find $(SRC) -name '*.cpp')

# List all of the o files from cu
CU_OBJS := $(shell find $(SRC) -name '*.cu' | sed -r "s/($(SRC))\/(.*)\.(cu)/obj\/\2\.obj/")

# List all of the o files from cpp
CPP_OBJS := $(shell find $(SRC) -name '*.cpp' | sed -r "s/($(SRC))\/(.*)\.(cpp)/obj\/\2\.obj/")

.PHONY: clean
.SILENT: $(BIN)/$(PLATFORM)/$(EXE) $(CU_OBJS) $(CPP_OBJS)$ clean format

obj/%.obj: $(SRC)/%.cu
	@mkdir -p $(@D)
	@echo -e "Make: compiling kernel file ${GREEN}$<${RESET}"
	@$(CC) -Iinclude $(CFLAGS) -maxrregcount=32 -o $@ -c $<
	
obj/%.obj: $(SRC)/%.cpp
	@mkdir -p $(@D)
	@echo -e "Make: compiling source file ${PINK}$<${RESET}"
	@$(CC) -Iinclude $(CFLAGS) -o $@ -c $<

all: $(BIN)/$(PLATFORM)/$(EXE)

$(BIN)/$(PLATFORM)/$(EXE): $(CU_OBJS) $(CPP_OBJS) 
	@echo -e "Make: building executable ${CYAN}$<${RESET}"
	@$(CC) -Llib/x64 -lglew64 $(CU_OBJS) $(CPP_OBJS) -o $(BIN)/$(EXE)
	
format:
	clang-format -i $(shell ls $(SRC)/*)

clean:
	@echo -e "Make: cleaning up"
	@rm -rf $(BIN)/$(PLATFORM)
	@rm -rf obj
