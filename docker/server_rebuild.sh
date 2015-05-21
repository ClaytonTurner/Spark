docker rm $(docker ps -aq)
sudo docker build -t server_container .
sudo docker run --name server_instance -m 256m -d -p 45001:11211 client_container
docker logs server_instance

