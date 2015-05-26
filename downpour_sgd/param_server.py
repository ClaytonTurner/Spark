'''
This file is for the param server which will host gradient descent updates
 to be read by and written to by the distbelief model replicas
'''

from socket import socket, gethostbyname, AF_INET, SOCK_DGRAM
import sys
PORT = 45001
SIZE = 1024 # May need to alter

hostname = gethostbyname('0.0.0.0')

my_socket = socket(AF_INET, SOCK_DGRAM)
my_socket.bind( (hostname, PORT) )

print "Test server listening on port {0}\n".format(PORT)

weights = []

while True:
	(data,addr) = my_socket.recvfrom(SIZE)
	# So we have received an update and it is stored in the variable data
	
	print data
sys.exit()
