# Drishti VOL Connector

### HDF5 Dependency

This VOL connector was tested with HDF5 1.14.

**Note**: Make sure you have ``libhdf5`` shared dynamic libraries. For Linux, it's ``libhdf5.so``, for OSX, it's ``libhdf5.dylib``.

### Generate HDF5 shared library

If you don't have the shared dynamic libraries, you'll need to reinstall HDF5:

- Get the latest version of the develop branch
- In the repo directory, run ``./autogen.sh``
- In your build directory, run configure and make sure you **DO NOT** have the option ``--disable-shared``, for example:
    > env CC=mpicc ../hdf5_dev/configure --enable-build-mode=debug --enable-internal-debug=all --enable-parallel
- make; make install

### Settings

To build Drishti HDF5 VOL connector:

>
    export HDF5_DIR=/path/to/your/hdf5/installation

    mkdir build
    cd build
    
    cmake ..
    
    make
    make install
