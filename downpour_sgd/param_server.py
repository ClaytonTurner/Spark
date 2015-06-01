'''
This file is for the param server which will host gradient descent updates
 to be read by and written to by the distbelief model replicas
'''

#from socket import socket, gethostbyname, AF_INET, SOCK_DGRAM
import socket
import sys
import threading
import SocketServer
from time import sleep
from SimpleXMLRPCServer import SimpleXMLRPCServer

HOST = socket.gethostname()
PORT = 45001
SIZE = 4096 # Find out how big our messages will be then fix this
	    # Or send a size first and handle that then
	    # TODO



'''
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
'''

def test_func(*nums):
	return sum(nums)

if __name__ == "__main__":
	# Port 0 means to select an arbitrary unused port
	#HOST, PORT = "localhost", 0 # defined earlier in file
	server = SimpleXMLRPCServer(("localhost",8000))
	print "Listening on port 8000..."
	server.register_function(test_func,"test function");
	server.serve_forever()

