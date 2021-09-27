CC          = ncc
CFLAGS      = -O3 -fopenmp -report-all -fdiag-vector=3 -fcse-after-vectorization  -msched-interblock


DEBUG	?=0
ifeq ($(DEBUG), 1)
CFLAGS	+=	-g -traceback=verbose
endif

BLAS		= -L/opt/nec/ve/nlc/2.3.0/lib/ -lblas_openmp -lsblas_openmp  -lcblas -fopenmp
LIBS		=  ${BLAS} ${LAPACK} -lasl_openmp

COM_SRCS := mmio.c sparse_matrix.c 
SRCS := par_iluc.c ${COM_SRCS}
OBJS := $(SRCS:%.c=%.o)

.PHONY = all clean
all: solver 

solver:	$(OBJS)
	${CC} ${CFLAGS} ${INC} $^ -o $@ ${LIBS}

%.o: %.c
	${CC} ${CFLAGS} ${INC} -c $<

clean:
	rm -rvf *.o solver 
	
