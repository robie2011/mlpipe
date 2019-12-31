vast create instance $1 --image tensorflow/tensorflow:2.0.0b1-gpu-py3-jupyter --disk 5

STATUS=$(vast show instances | tail -n1 | tr -s " " | cut -d' ' -f3)

while [ $STATUS -ne "running" ]
do
  sleep 5s
  STATUS=$(vast show instances | tail -n1 | tr -s " " | cut -d' ' -f3)
  echo "check $(date) ..."
done

echo "container is running!"