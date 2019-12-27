DESC_INSTANCES=$(vast show instances | tail -n1)
export SSH_PORT=$(echo $DESC_INSTANCES| tr -s " "| cut -d' ' -f14)
export SSH_HOST=$(echo $DESC_INSTANCES| tr -s " "| cut -d' ' -f13)
export SSH_HOST_USER=root

if [ -z "$SSH_HOST" ]; then
  echo "no instance found"
  exit
fi
