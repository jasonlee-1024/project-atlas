SDK_INSTALL_PATH :=   /usr/local/cuda-12.4
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := YourNameCUDA1.tar
EXECS :=  q1 q2 q3
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt
#######################################################################

timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#######################################################################
q1 : q1.cpp
	g++ $< -o $@ $(OPTIONS)

q2 : q2.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)

q3 : q3.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)






