#!/bin/sh
echo "Running script"
# run applications 
python -u /tmp/spark/client.py # -u flag unbuggers output so we can capture it in logs
 
exit 0

