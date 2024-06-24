#!/bin/bash
set -e

if [ -z $1 ] || [ -z $2 ]; then
    echo "Usage: $0 <username> <hostfile>"
    exit 1
fi

HOME_ROOT=${HOME_ROOT:-/root/sharedDisk/home}

username=$1
hostfile=$2

THIS_DIR=$(dirname $0)
PASSWD=$(openssl passwd hpcai)

useradd -d $HOME_ROOT/$username -m -N -g train-users -p "$PASSWD" -f 1 -s /bin/bash $username
python3 $THIS_DIR/sync-users.py $hostfile