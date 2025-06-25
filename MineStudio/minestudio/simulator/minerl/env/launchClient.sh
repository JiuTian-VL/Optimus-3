#!/bin/bash

replaceable=0
port=0
seed="NONE"
maxMem="2G"
device="egl"
fatjar=build/libs/mcprec-6.13.jar

while [ $# -gt 0 ]
do
    case "$1" in
        -replaceable) replaceable=1;;
        -port) port="$2"; shift;;
        -seed) seed="$2"; shift;;
        -maxMem) maxMem="$2"; shift;;
        -device) device="$2"; shift;;
        -fatjar) fatjar="$2"; shift;;
        *) echo >&2 \
            "usage: $0 [-replaceable] [-port <port>] [-seed <seed>] [-maxMem <maxMem>] [-device <device>] [-fatjar <fatjar>]"
            exit 1;;
    esac
    shift
done
  
if ! [[ $port =~ ^-?[0-9]+$ ]]; then
    echo "Port value should be numeric"
    exit 1
fi


if [ \( $port -lt 0 \) -o \( $port -gt 65535 \) ]; then
    echo "Port value out of range 0-65535"
    exit 1
fi

if [ "$device" == "cpu" ]; then
    xvfb-run -a java -Xmx$maxMem -jar $fatjar --envPort=$port
else
    vglrun -d $device java -Xmx$maxMem -jar $fatjar --envPort=$port
fi

[ $replaceable -gt 0 ]

