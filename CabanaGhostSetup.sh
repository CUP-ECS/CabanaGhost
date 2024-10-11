#!/bin/bash

setup() {
    echo "Setting up directories..."

    # Check if directories exist
    if [ -d $HOME/repos ]; then
        echo "'repos' directory already exists."
    else
        mkdir $HOME/repos
    fi

    if [ -d $HOME/install ]; then
        echo "'install' directory already exists."
    else
        mkdir $HOME/install; 
        mkdir $HOME/install/kokkos $HOME/install/Cabana $HOME/install/Silo; 
    fi

    # Kokkos setup
    # Check if Kokkos is there
    if [ -d $HOME/repos/kokkos ]; then
        echo "'kokkos' directory already exists."
    else
        git clone git@github.com:kokkos/kokkos.git $HOME/repos/kokkos -j16
    fi

    if [ -d $HOME/repos/kokkos/build ]; then
        rm -rf $HOME/repos/kokkos/build; mkdir $HOME/repos/kokkos/build
    else
        mkdir $HOME/repos/kokkos/build
    fi

    cd $HOME/repos/kokkos
    export CRAYPE_LINK_TYPE=dynamic
    # install ROCm?
    # -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON
    cmake -S $HOME/repos/kokkos -B $HOME/repos/kokkos/build -DCMAKE_INSTALL_PREFIX=$HOME/install/kokkos -DCMAKE_BUILD_TYPE=Release
    cd build; make -j16; make install -j16
    echo "Kokkos setup complete!"

    # Cabana setup
    # Check if Cabana is there
    if [ -d $HOME/repos/Cabana ]; then
        echo "'Cabana' directory already exists."
    else
        git clone https://github.com/ECP-copa/Cabana.git $HOME/repos/Cabana -j16
    fi

    export KOKKOS_INSTALL_DIR=$HOME/install/kokkos
    export CABANA_DIR=$HOME/install/Cabana

    if [ -d $HOME/repos/Cabana/build ]; then
        rm -rf $HOME/repos/Cabana/build; mkdir $HOME/repos/Cabana/build
    else
        mkdir $HOME/repos/Cabana/build
    fi

    cd $HOME/repos/Cabana/build
    cmake \
	    -D CMAKE_BUILD_TYPE="Release" \
	    -D CMAKE_PREFIX_PATH=$KOKKOS_INSTALL_DIR \
	    -D CMAKE_INSTALL_PREFIX=$CABANA_DIR \
	    -D Cabana_ENABLE_GRID=ON \
	    -D Cabana_ENABLE_MPI=ON \
	    ..;
    make install -j16
    echo "Cabana setup complete!"

    # Silo setup
    if [ -d $HOME/repos/Silo ]; then
        echo "'Silo' directory already exists."
    else
        git clone git@github.com:LLNL/Silo.git $HOME/repos/Silo -j16
    fi

    cd $HOME/repos/Silo; export SILO_INSTALL_DIR=$HOME/install/Silo
    ./configure --prefix=$SILO_INSTALL_DIR
    make -j16; make install -j16
    echo "Silo setup complete!"

    # CabanaGhost setup
    if [ -d $HOME/repos/CabanaGhost ]; then
        echo "'CabanaGhost' directory already exists"
    else
	# clones into blt branch and downloads the blt module
        git clone -b blt --recurse-submodules -j16 git@github.com:CUP-ECS/CabanaGhost.git $HOME/repos/CabanaGhost
    fi

    cd $HOME/repos/CabanaGhost

    # Update the CMakeLists.txt for CabanaGhost to use the correct blt path
    # TODO verify if this is necessary
    # sed -i "s|include(blt/SetupBLT.cmake)|include($HOME/repos/blt/SetupBLT.cmake)|" $HOME/repos/CabanaGhost/CMakeLists.txt
    
    if [ -d $HOME/repos/CabanaGhost/build ]; then
        rm -rf build; mkdir build; cd build
    else
        mkdir build; cd build
    fi

    cmake -DBLT_CXX_STD=c++14 -D CMAKE_PREFIX_PATH="$KOKKOS_INSTALL_DIR;$CABANA_DIR;$SILO_INSTALL_DIR" ..
    make -j16
    echo "CabanaGhost setup complete!"

    echo "Setup complete!"
}

test() {
    cd $HOME/repos/CabanaGhost/tests
    cmake ..
}

clean() {
    echo "Cleaning up..."
    rm -rf $HOME/install $HOME/repos/Silo $HOME/repos/kokkos $HOME/repos/Cabana
    rm -rf $HOME/repos/CabanaGhost/build
    echo "Cleanup complete!"
}

# Check for arguments
if [ "$1" == "clean" ]; then
    clean
elif [ "$1" == "test" ]; then
    test
else
    setup
fi
