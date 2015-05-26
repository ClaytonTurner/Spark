if [ $# -eq 1 ] 
	then
		con_type=$1
		docker rm $(docker ps -aq)
		sudo docker build -t ${con_type}_container .
		sudo docker run --name ${con_type}_instance -m 256m -d -p 45001:11211 ${con_type}_container
		docker logs ${con_type}_instance
	else
		echo "Incorrect usage. sh rebuild.sh <name_type>"
fi

