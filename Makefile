EXE_NAME = enigma.exe
BIN_DIR = bin 
DEVICE_CC = nvcc
DEVICE_LD = nvcc
HOST_CC = cl
HOST_LD = link
OBJ_DIR = obj

DEVICE_CC_FLAGS = --gpu-architecture=sm_50 
DEVICE_LD_FLAGS = --gpu-architecture=sm_50 
HOST_CC_FLAGS = -nologo
HOST_LD_FLAGS = -nologo

CUDA_SRCS = *.cu
C_SRCS = *.c
CUDALINK = $(OBJ_DIR)\cuda_link.obj

CUDA_OBJ = $(OBJ_DIR)\enigma.obj $(OBJ_DIR)\main.obj
C_OBJ = $(OBJ_DIR)\preprocessing.obj 

.PHONY: all 
.SUFFIXES : .cu
.SILENT :

all: $(EXE_NAME)

{}.cu{$(OBJ_DIR)\}.obj:
	@if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)
	$(DEVICE_CC) -dc $< -o $@ $(DEVICE_CC_FLAGS)
	
	
{}.c{$(OBJ_DIR)\}.obj:
	@if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)
	echo $(C_OBJ)
	$(HOST_CC) /c /Fo.\$@ $< $(HOST_CC_FLAGS)
	
	
$(CUDALINK): $(CUDA_OBJ) #$(OBJ_DIR)$(CUDA_OBJ)
	$(DEVICE_LD) --device-link $(CUDA_OBJ) --output-file $(CUDALINK) $(DEVICE_LD_FLAGS) 
	
$(EXE_NAME): $(CUDALINK) $(C_OBJ)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(HOST_LD) $(CUDA_OBJ) $** /OUT:$(BIN_DIR)\$@ /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\lib\x64" /DEFAULTLIB:"cudart" $(HOST_LD_FLAGS)


clean:
	@echo "cleaning..."
	@if exist $(BIN_DIR) rmdir /S /Q $(BIN_DIR) 
	@if exist $(OBJ_DIR) rmdir /S /Q $(OBJ_DIR)
	

