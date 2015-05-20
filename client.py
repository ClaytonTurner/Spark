import sys
from socket import socket, AF_INET, SOCK_DGRAM

#IP_SERVER = "192.168.137.51"
IP_SERVER = "..."
#PORT = 5000
PORT = 45001
SIZE = 1024
print "Test client sending packets to IP {0}, via port {1}\n".format(IP_SERVER, PORT)

my_socket = socket(AF_INET, SOCK_DGRAM)

while True:
	my_socket.sendto('cool',(IP_SERVER,PORT))
sys.exit()
