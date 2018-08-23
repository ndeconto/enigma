EXE_NAME = enigma.exe
BIN_DIR = bin 
DEVICECC = nvcc
DEVICELD = nvcc
HOSTCC = cl
HOSTLD = link
ODIR = obj

DEVICECCFLAGS = --gpu-architecture=sm_50
DEVICELDFLAGS = --gpu-architecture=sm_50
HOSTCCFLAGS =

CUDASRCS = enigma.cu main.cu
CSRCS = preprocessing.c
CUDALINK = $(ODIR)\cuda_link.obj

CUDAOBJ = obj\enigma.obj obj\main.obj
#CUDAOBJ = $(CUDASRCS:.cu=.obj) #$(patsubst %.cu, $(ODIR)%.obj, $(CUDASRCS))
COBJ = obj\preprocessing.obj #$(CSRCS:.c=.obj) #$(patsubst %.c, $(ODIR)%.obj, $(CSRCS))
#COBJ = {CSRCS:%.c=lol%.obj}

.PHONY: all 
.SUFFIXES : .cu

all: $(EXE_NAME)

{}.cu{$(ODIR)\}.obj:
	@if not exist $(ODIR) mkdir $(ODIR)
	$(DEVICECC) -dc $< -o $@ $(DEVICECCFLAGS)
	
	
{}.c{$(ODIR)\}.obj:
	@if not exist $(ODIR) mkdir $(ODIR)
	echo $(COBJ)
	$(HOSTCC) /c /Fo.\$@ $< $(HOSTCCFLAGS)
	
	
$(CUDALINK): $(CUDAOBJ) #$(ODIR)$(CUDAOBJ)
	$(DEVICELD) --device-link $(CUDAOBJ) --output-file $(CUDALINK) $(DEVICELDFLAGS) 
	
$(EXE_NAME): $(CUDALINK) $(COBJ)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(HOSTLD) $(CUDAOBJ) $** /OUT:$(BIN_DIR)\$@ /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\lib\x64" /DEFAULTLIB:"cudart"


clean:
	@echo "cleaning..."
	@if exist $(BIN_DIR) rmdir /S /Q $(BIN_DIR) 
	@if exist $(ODIR) rmdir /S /Q $(ODIR)
	

