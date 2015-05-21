############################################################
# Dockerfile to run Memcached Containers
# Based on Ubuntu Image
############################################################

# Set the base image to use to Ubuntu
#FROM ubuntu
FROM phusion/baseimage:0.9.16

# Set the file maintainer (your name - the file's author)
MAINTAINER Clayton Turner <caturner3@g.cofc.edu>

# Update the default application repository sources list
RUN apt-get update

# Installs
RUN apt-get -y install python

# Port to expose (default: 11211)
EXPOSE 11211

# Default Memcached run command arguments
#CMD ["-u", "root", "-m", "128"]
#CMD ["/sbin/my_init"]

# Clean up APT when done - maybe do this at the end, but we're storing stuff in /tmp/*
RUN sudo apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add my stuff
#RUN mkdir -p /etc/my_init.d
#ADD client.sh /etc/my_init.d/client.sh
#ADD client.py /usr/local/sbin/client.py
COPY docker_launch.sh /tmp/spark/
COPY client.py /tmp/spark/

# Set the user to run Memcached daemon
USER root
ADD sudoers.txt /etc/sudoers
RUN chmod 440 /etc/sudoers
USER daemon

# Set the entrypoint to memcached binary
#ENTRYPOINT memcached
RUN sudo sh /tmp/spark/docker_launch.sh
RUN echo "End of Dockerfile"

