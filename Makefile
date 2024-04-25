CC=mpicc
AR=ar

DEBUG=-DENABLE_DRISHTI_LOGGING -g -O0
INCLUDES=-I$(HDF5_DIR)/include
CFLAGS = $(DEBUG) -fPIC $(INCLUDES) -Wall
LIBS=-L$(HDF5_DIR)/lib -lhdf5 -lz
# Uncomment this line Linux builds:
# DYNLDFLAGS = $(DEBUG) -shared -fPIC $(LIBS)
# Uncomment this line MacOS builds:
DYNLDFLAGS = $(DEBUG) -dynamiclib -current_version 1.0 -fPIC $(LIBS)
LDFLAGS = $(DEBUG) $(LIBS)
ARFLAGS = rs

DYNSRC = H5VLdrishti.c
DYNOBJ = $(DYNSRC:.c=.o)
# Uncomment this line Linux builds:
# DYNLIB = libh5drishti_vol.so
# Uncomment this line MacOS builds:
DYNLIB = libh5drishti_vol.dylib
DYNDBG = libh5drishti_vol.dylib.dSYM

#STATSRC = new_h5api.c
#STATOBJ = $(STATSRC:.c=.o)
#STATLIB = libnew_h5api.a

EXSRC = example.c
EXOBJ = $(EXSRC:.c=.o)
EXEXE = example.exe
EXDBG = example.exe.dSYM

ASYNC_EXSRC = async_new_h5api_ex.c
ASYNC_EXOBJ = $(ASYNC_EXSRC:.c=.o)
ASYNC_EXEXE = async_new_h5api_ex.exe
ASYNC_EXDBG = async_new_h5api_ex.exe.dSYM

DATAFILE = testfile.h5

all: $(EXEXE) $(ASYNC_EXEXE) $(DYNLIB) $(STATLIB)

$(EXEXE): $(EXSRC) $(STATLIB) $(DYNLIB)
	$(CC) $(CFLAGS) $^ -o $(EXEXE) $(LDFLAGS)

$(ASYNC_EXEXE): $(ASYNC_EXSRC) $(STATLIB) $(DYNLIB)
	$(CC) $(CFLAGS) $^ -o $(ASYNC_EXEXE) $(LDFLAGS)

$(DYNLIB): $(DYNSRC)
	$(CC) $(CFLAGS) $(DYNLDFLAGS) $^ -o $@

$(STATOBJ): $(STATSRC)
	$(CC) -c $(CFLAGS) $^ -o $(STATOBJ)

$(STATLIB): $(STATOBJ)
	$(AR) $(ARFLAGS) $@ $^

.PHONY: clean all
clean:
	rm -rf $(DYNOBJ) $(DYNLIB) $(DYNDBG) \
            $(STATOBJ) $(STATLIB) \
            $(EXOBJ) $(EXEXE) $(EXDBG) \
            $(ASYNC_EXOBJ) $(ASYNC_EXEXE) $(ASYNC_EXDBG) \
            $(DATAFILE)
