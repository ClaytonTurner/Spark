from socket import socket, gethostbyname, AF_INET, SOCK_DGRAM
import sys

'''
Data type (i.e. parameters or gradients) is dependent upon port
45001 : parameters
45002 : gradients
'''


def send_data(data, PORT, IP):
	# Send data to location provided in loc (IP)
	SIZE = 1024 # May need to alter
	hostname = gethostbyname('0.0.0.0')

	my_socket = socket(AF_INET, SOCK_DGRAM)
	my_socket.bind( (hostname, PORT) )

	while True:
		my_socket.sendto(data,(IP,PORT))

	#print "Test server listening on port {0}\n".format(PORT)
	return True # Signifies no error

def receive_data(PORT):
	SIZE = 1024
	hostname = gethostbyname('0.0.0.0')

	my_socket = socket(AF_INET, SOCK_DGRAM)
	my_socket.bind( (hostname, PORT) )

	data = None
	while data is not None:
		# Should probably add a timeout
		(data,addr) = my_socket.recvfrom(SIZE)
	return data # Signifies no error
