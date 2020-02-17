vast show instances --raw>/tmp/vast.json
export SSH_HOST=$(cat /tmp/vast.json | json "[0].ssh_host")
export SSH_PORT=$(cat /tmp/vast.json | json "[0].ssh_port")
export SSH_HOST_USER=root

if [ -z "$SSH_HOST" ]; then
  echo "no instance found"
  exit
fi
