#!/bin/bash
ROOT_DIR=~

## https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/releases/tag/2.4.0
cd $ROOT_DIR
wget -c https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/archive/refs/tags/2.4.0.tar.gz
tar -zxf 2.4.0.tar.gz && cd SPlisHSPlasH-2.4.0
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_PYTHON_BINDINGS=Off ..
make -j 4

## https://github.com/wdas/partio.git^c869a118bc7b11e7d92ed86d25959ef09c9ae6ae
cd $ROOT_DIR
git clone https://github.com/wdas/partio.git && cd partio
git checkout c869a118bc7b11e7d92ed86d25959ef09c9ae6ae
prefix=$HOME/local
make -j prefix=$prefix install

## OSError: libnvidia-ml.so: cannot open shared object file: No such file or directory
ln -s libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so

## Optional
## https://github.com/Blosc/c-blosc/archive/refs/tags/v1.5.0.zip
cd $ROOT_DIR
wget -c https://github.com/Blosc/c-blosc/archive/refs/tags/v1.5.0.zip
unzip v1.5.0.zip && cd c-blosc-1.5.0
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make && make install
## https://sourceforge.net/projects/boost/files/boost/1.70.0/
wget -c https://cytranet.dl.sourceforge.net/project/boost/boost/1.70.0/boost_1_70_0.tar.gz
tar -zxf boost_1_70_0.tar.gz && cd boost_1_70_0
./bootstrap.sh --with-python=$(which python3)
./b2 --with-python --with-iostreams --with-system --with-regex install
ldconfig
## https://github.com/AcademySoftwareFoundation/openvdb/releases/tag/v7.1.0
cd $ROOT_DIR
wget -c https://github.com/AcademySoftwareFoundation/openvdb/archive/refs/tags/v7.1.0.tar.gz
tar -zxf v7.1.0.tar.gz && cd openvdb-7.1.0
mkdir build
cd build
cmake -DOPENVDB_BUILD_PYTHON_MODULE=ON -DUSE_NUMPY=ON -DOPENVDB_PYTHON_WRAP_ALL_GRID_TYPES=ON -DPYTHON_EXECUTABLE='/usr/bin/python3.6' -DCMAKE_CXX_FLAGS="-I/usr/local/lib/python3.6/dist-packages/numpy/core/include -I/usr/local/lib/python3.6/dist-packages/numpy/core/include/numpy" ..
make -j$(nproc)
make install
