#!/bin/sh
# '/sbin/setuser memcache' runs the given command as the user 'memcache'
# If you omit that part, the command will be run as root
# Make sure this file is chmod +x
#exec /sbin/setuser memcache /usr/bin/memcached >>/var/log/memcached.log 2>&1
case "$1" in
  start)
    echo "Running script"
    # run application you want to start
    python /usr/local/sbin/client.py &
    ;;
  stop)
    echo "Stopping script"
    # kill application you want to stop
    killall python
    ;;
  *)
    echo "Usage: /etc/init.d/example{start|stop}"
    exit 1
    ;;
esac
 
exit 0

