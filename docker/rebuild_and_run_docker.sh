docker rm $(docker ps -aq)
sudo docker build -t client_container .
sudo docker run --name client_instance -m 256m -d -p 45001:11211 client_container
docker logs client_instance

