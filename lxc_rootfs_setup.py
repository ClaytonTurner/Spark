import os
import sys

# This will only work on our linux cluster
# You may need to run this with sudo or change permissions accordingly

loc = "/var/lib/lxc"
for spark_vm in os.listdir(loc):
	if spark_vm.split("6")[0] == "spark":
		print loc+"/"+spark_vm+"/fstab"
		f = open(loc+"/"+spark_vm+"/fstab","w")
		f.write(fstab_str)
		f.close()


