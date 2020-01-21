#!/bin/bash

if [[ -z $1 ]]; then
	echo "usage: $1 <install-prefix>"
	exit 1
fi

wget https://github.com/xianyi/OpenBLAS/archive/v0.3.6.tar.gz
tar xzf v0.3.6.tar.gz
cd OpenBLAS-0.3.6
make
make PREFIX=$1 install
