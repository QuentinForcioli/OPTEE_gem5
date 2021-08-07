#!/bin/sh
#
# Start a script from m5 readfile
#

case "$1" in
  start)
    echo "Launching m5 bash script:"
    m5 readfile | sh;
    echo "m5 bash script finished"
    ;;
  stop)
    ;;
  restart|reload)
    ;;
  *)
    echo $"Usage: $0 {start|stop|restart}"
    exit 1
esac

exit $?