#!/bin/bash

setup() {
    module load rocm/6.1.2
    init
    silo
    kokkos
    cabana
    cabanaghost
    printf "Setup complete!\n"
}

init() {
    printf "Setting up directories...\n"
    if [ -d $HOME/repos ]; then
        :
    else
        mkdir $HOME/repos
    fi

    if [ -d $HOME/install ]; then
        :
    else
        mkdir $HOME/install; 
        mkdir $HOME/install/kokkos $HOME/install/Cabana $HOME/install/Silo; 
    fi
    printf "Complete!\n"
}

kokkos() {
    module load rocm/6.1.2 # duplicated in case setup() is not called
    if [ -d $HOME/repos/kokkos ]; then
        printf "'kokkos' directory already exists."
    else
        git clone git@github.com:kokkos/kokkos.git $HOME/repos/kokkos -j16
    fi

    if [ -d $HOME/repos/kokkos/build ]; then
        rm -rf $HOME/repos/kokkos/build; mkdir $HOME/repos/kokkos/build
    else
        mkdir $HOME/repos/kokkos/build
    fi

    cd $HOME/repos/kokkos
    export CRAYPE_LINK_TYPE=dynamic # Necessary?
    cmake \
        -S $HOME/repos/kokkos \
        -B $HOME/repos/kokkos/build \
        -DCMAKE_CXX_COMPILER=hipcc \
        -DCMAKE_INSTALL_PREFIX=$HOME/install/kokkos \
        -DKokkos_ENABLE_HIP=ON \
        -DKokkos_ARCH_AMD_GFX90A=ON \
        -DCMAKE_BUILD_TYPE=Release;
    cd build; make -j16; make install -j16
    printf "Kokkos setup complete!\n\n"
}

silo() {
    module load rocm/6.1.2
    if [ -d $HOME/repos/Silo ]; then
        printf "'Silo' directory already exists."
    else
        git clone git@github.com:LLNL/Silo.git $HOME/repos/Silo -j16
    fi

    cd $HOME/repos/Silo;
    ./configure --prefix=$HOME/install/Silo # possibly replace with cmake
    make -j16; make install -j16
    printf "Silo setup complete!\n\n"
}

cabana() {
    module load rocm/6.1.2 # duplicated in case setup() is not called
    if [ -d $HOME/repos/Cabana ]; then
        printf "'Cabana' directory already exists."
    else
        git clone https://github.com/ECP-copa/Cabana.git $HOME/repos/Cabana -j16
    fi

    if [ -d $HOME/repos/Cabana/build ]; then
        rm -rf $HOME/repos/Cabana/build; mkdir $HOME/repos/Cabana/build
    else
        mkdir $HOME/repos/Cabana/build
    fi

    cd $HOME/repos/Cabana/build
    cmake \
	    -D CMAKE_BUILD_TYPE="Release" \
        -DCMAKE_CXX_COMPILER=hipcc \
	    -D CMAKE_PREFIX_PATH="$HOME/install/kokkos;$HOME/install/Silo" \
	    -D CMAKE_INSTALL_PREFIX=$HOME/install/Cabana \
	    -D Cabana_ENABLE_GRID=ON \
	    -D Cabana_ENABLE_MPI=ON \
	    ..;
    make install -j16
    printf "Cabana setup complete!\n\n"
}

cabanaghost() {
    module load rocm/6.1.2 # duplicated in case setup() is not called
    if [ -d $HOME/repos/CabanaGhost ]; then
        :
    else
	# clones into blt branch and downloads the blt module
        git clone -b blt --recurse-submodules -j16 git@github.com:CUP-ECS/CabanaGhost.git $HOME/repos/CabanaGhost
    fi

    cd $HOME/repos/CabanaGhost
    
    if [ -d $HOME/repos/CabanaGhost/build ]; then
        rm -rf build; mkdir build; cd build
    else
        mkdir build; cd build
    fi

    cmake \
        -DBLT_CXX_STD=c++20 \
        -DCMAKE_C_COMPILER=cc \
        -DCMAKE_CXX_COMPILER=hipcc \
        -D CMAKE_PREFIX_PATH="$HOME/install/kokkos;$HOME/install/Cabana;$HOME/install/Silo" \
        ..;
    make -j16
    printf "CabanaGhost setup complete!\n"
}

test() {
    module load rocm/6.1.2 # duplicated in case setup() is not called
    cd $HOME/repos/CabanaGhost/tests
    cmake \
        -DBLT_CXX_STD=c++20 \
        -DCMAKE_C_COMPILER=cc \
        -DCMAKE_CXX_COMPILER=hipcc \
        -D CMAKE_PREFIX_PATH="$HOME/install/kokkos;$HOME/install/Cabana;$HOME/install/Silo" \
        -D CMAKE_MODULE_PATH="$HOME/install/kokkos;$HOME/install/Cabana;$HOME/install/Silo" \
        ..;
    make all;
    cd $HOME/repos/CabanaGhost/build; make; cd $HOME/repos/CabanaGhost/build/tests; make;
    blt_gtest_smoke;
}

clean() {
    printf "Cleaning up...\n"
    rm -rf $HOME/install/kokkos $HOME/install/Cabana $HOME/install/Silo
    rm -rf $HOME/repos/Silo $HOME/repos/kokkos $HOME/repos/Cabana
    rm -rf $HOME/repos/CabanaGhost/build
    printf "Cleanup complete!\n"
}

# Check for arguments
if [ "$1" == "clean" ]; then
    clean
elif [ "$1" == "test" ]; then
    test
elif [ "$1" == "init" ]; then
    init
elif [ "$1" == "kokkos" ]; then
    kokkos
elif [ "$1" == "silo" ]; then
    silo
elif [ "$1" == "cabana" ]; then
    cabana
elif [ "$1" == "cabanaghost" ]; then
    cabanaghost
else
    setup
fi
