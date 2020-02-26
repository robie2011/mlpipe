#vast create instance $1 --image tensorflow/tensorflow:2.0.0b1-gpu-py3-jupyter --disk 5
vast create instance $1 --image robie2011/mlpipe --disk 5

STATUS=$(vast show instances | tail -n1 | tr -s " " | cut -d' ' -f3)

while [ "$STATUS" != "running" ]
do
  sleep 5s
  STATUS=$(vast show instances | tail -n1 | tr -s " " | cut -d' ' -f3)
  echo "check $(date): $STATUS ..."
done

echo "container is running!"