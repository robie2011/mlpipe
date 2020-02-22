vast show instances --raw>/tmp/vast.json


if [ -z "$1" ]
then
  echo "use first entry"
  obj=$(cat /tmp/vast.json | json "[0]")
else
    echo "set id is: $1"
    obj=$(cat /tmp/vast.json | json -c "this.id == $1" "[0]")
fi



export SSH_HOST=$(echo $obj | json "ssh_host")
export SSH_PORT=$(echo $obj | json "ssh_port")
export SSH_HOST_USER=root

if [ -z "$SSH_HOST" ]; then
  echo "no instance found"
  exit
fi
