#!/bin/bash

# This file is for testing downpour sgd locally


for i in `seq 1 5`;
do
	echo $i
	python downpour_sgd_client.py &
done
