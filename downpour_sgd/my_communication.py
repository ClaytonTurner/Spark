from socket import socket, gethostbyname, AF_INET, SOCK_DGRAM
import sys


def send_data(data, loc):
	# Send data to location provided in loc (IP)
	PORT = loc
	SIZE = 1024 # May need to alter
	hostname = gethostbyname('0.0.0.0')

	my_socket = socket(AF_INET, SOCK_DGRAM)
	my_socket.bind( (hostname, PORT) )

	#print "Test server listening on port {0}\n".format(PORT)
	return True # Signifies no error

def receive_data():
	data = None
	while data is not None:
		# Should probably add a timeout
		(data,addr) = my_socket.recvfrom(SIZE)
	return data # Signifies no error
