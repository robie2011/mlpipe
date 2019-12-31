DESC_INSTANCES=$(vast show instances | tail -n1 | tr -s " ")
export SSH_HOST=$(echo $DESC_INSTANCES | cut -d' ' -f12)
export SSH_PORT=$(echo $DESC_INSTANCES | cut -d' ' -f13)
export SSH_HOST_USER=root

if [ -z "$SSH_HOST" ]; then
  echo "no instance found"
  exit
fi
