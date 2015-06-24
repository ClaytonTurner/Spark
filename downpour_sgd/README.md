## Downpour_Sgd_Client
The clients exist to each host a replica - similar to a thread
Each client will get 1/nth of the data assuming n clients

## Param_Server
You must specify the amount of clients in the param server in order to correctly shard the data
The param server exists on 192.168.137.62 - hardcoding is fine here
The param server listens on port 49151 (non-registered port)   

## Replica
Each replica is based off of the Distbelief model
Each replica will be running an adagrad-based neural network

##### Distbelief
