import os
import sys

# This will only work on our linux cluster
# You may need to run this with sudo or change permissions accordingly

loc = "/var/lib/lxc"
for d_sgd_vm in os.listdir(loc):
	if d_sgd_vm.split("_sgd")[0] == "downpour":
		print loc+"/"+d_sgd_vm+"/fstab"
		f = open(loc+"/"+d_sgd_vm+"/fstab","w")
		f.write(fstab_str)
		f.close()


