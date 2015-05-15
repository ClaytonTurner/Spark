from socket import socket, gethostbyname, AF_INET, SOCK_DGRAM
import sys
PORT = 5000
SIZE = 1024

hostname = gethostbyname('0.0.0.0')

my_socket = socket(AF_INET, SOCK_DGRAM)
my_socket.bind( (hostname, PORT) )

print "Test server listening on port {0}\n".format(PORT)

while True:
	(data,addr) = my_socket.recvfrom(SIZE)
	print data
sys.exit()
