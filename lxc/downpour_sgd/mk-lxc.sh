#!/bin/bash
if [[ $EUID != 0 ]]; then
	echo "Must run as root, attempting to sudo ..."
	sudo $0 $*
	exit $?
fi

if [[ $# != 1 ]]; then
	echo "Usage: $0 NEXT_LETTER"
	echo "Where you have to keep track of NEXT_LETTER!"
	exit 1
fi

NEXT_LETTER=$1
MNAME=downpour_sgd$NEXT_LETTER

lxc-create -t debian -n $MNAME
cat << EOF >>/var/lib/lxc/$MNAME/config
# cgroups
lxc.cgroup.cpu.shares = 128
lxc.cgroup.memory.soft_limit_in_bytes = 4000000000
lxc.cgroup.memory.limit_in_bytes = 8000000000
lxc.cgroup.blkio.weight = 250
EOF
echo "Created machine and fixed configuration."
echo "Log in with login root/root, then run passwd"
echo "And use ifconfig and note the IP for the machine"
#lxc-execute -n downpour_sgd$NEXT_LETTER sh ./run_on_lxc_creation.sh
lxc-start -n $MNAME

