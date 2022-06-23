FROM ubuntu:focal

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
  wget \
  make \
  cmake \
  gcc \
  git \
  vim \
  libeigen3-dev \
  python3-pip \
  libboost-all-dev \
  libhdf5-dev \
  libssl-dev \
  libxml2-dev \
  libpciaccess-dev \
  libopenblas-dev \
  liblapack-dev \
  python3-pybind11
RUN pip3 install --upgrade pip

# Install MKL
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB -O - | apt-key add -
RUN echo "deb https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update && apt-get -y install intel-oneapi-mkl intel-oneapi-mkl-devel
RUN ln -r -s /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_avx2.so.2 /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_avx2.so
RUN ln -r -s /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_vml_avx2.so.2 /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_vml_avx2.so
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:$LD_LIBRARY_PATH
ENV MKLROOT=/opt/intel/oneapi/mkl/latest

# Install Cython
RUN pip3 install Cython pythran

# Install numpy
RUN git clone https://github.com/numpy/numpy.git numpy
RUN cd numpy ; \
  cp site.cfg.example site.cfg ; \
  echo "\n[mkl]" >> site.cfg ; \
  echo "include_dirs = /opt/intel/oneapi/mkl/latest/lib/intel64/" >> site.cfg ; \
  echo "library_dirs = /opt/intel/oneapi/mkl/latest/lib/intel64/" >> site.cfg ; \
  echo "mkl_libs = mkl_rt" >> site.cfg ; \
  echo "lapack_libs =" >> site.cfg
RUN cd numpy ; git submodule update --init
RUN cd numpy ; python3 setup.py build --fcompiler=gnu95
RUN cd numpy ; python3 setup.py install
RUN rm -rf numpy

# Install scipy
RUN git clone https://github.com/scipy/scipy.git scipy
RUN cd scipy ; git submodule update --init
RUN cd scipy ; python3 setup.py build
RUN cd scipy ; python3 setup.py install
RUN rm -rf scipy

# Install MongoDB C++ bindings
RUN wget https://github.com/mongodb/mongo-c-driver/releases/download/1.17.3/mongo-c-driver-1.17.3.tar.gz
RUN tar -xzf mongo-c-driver-1.17.3.tar.gz
RUN mkdir -p mongo-c-driver-1.17.3/cmake-build
RUN cd mongo-c-driver-1.17.3/cmake-build ; cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_AUTOMATIC_INIT_AND_CLEANUP=OFF -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_PREFIX_PATH=/usr ..
RUN cd mongo-c-driver-1.17.3/cmake-build ;make install

RUN wget https://github.com/mongodb/mongo-cxx-driver/archive/r3.6.2.tar.gz
RUN tar -xzf r3.6.2.tar.gz
RUN mkdir -p mongo-cxx-driver-r3.6.2/build
RUN cd mongo-cxx-driver-r3.6.2/build ; cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_VERSION=3.6.2 -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_PREFIX_PATH=/usr ..
RUN cd mongo-cxx-driver-r3.6.2/build ; make install

# Install Puffin
# TODO change url
RUN git clone https://github.com/qcscine/puffin.git
RUN cd puffin ; pip3 install .

# Bootstrap Puffin
ENV PUFFIN_PROGRAMS_XTB_AVAILABLE=true
RUN python3 -m scine_puffin bootstrap
RUN mv puffin.sh /scratch/puffin.sh
RUN chmod u+x /scratch/puffin.sh

ENTRYPOINT source ./puffin.sh ; python3 -m scine_puffin container
