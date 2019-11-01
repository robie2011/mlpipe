import sys
import os
import math
from confluent_kafka import Producer
from random import random
import time
import json
from datetime import datetime

if __name__ == '__main__':
    topic = "7msxg02r-random_sensors"
    #topic = os.environ['CLOUDKARAFKA_TOPIC']

    # Consumer configuration
    # See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
    conf = {
        'bootstrap.servers': os.environ['CLOUDKARAFKA_BROKERS'],
        'session.timeout.ms': 6000,
        'default.topic.config': {'auto.offset.reset': 'smallest'},
        # 'security.protocol': 'SASL_SSL',
	    # 'sasl.mechanisms': 'SCRAM-SHA-256',
        # 'sasl.username': os.environ['CLOUDKARAFKA_USERNAME'],
        # 'sasl.password': os.environ['CLOUDKARAFKA_PASSWORD']
    }

    p = Producer(**conf)

    def delivery_callback(err, msg):
        if err:
            sys.stderr.write('%% Message failed delivery: %s\n' % err)
        else:
            # sys.stderr.write('%% Message delivered to %s [%d]\n' %
            #                  (msg.topic(), msg.partition()))
            print(msg.value())

    while True:
        try:
            sensors = [
                int(random() * 100),
                200 + int(random() * 100),
                1000 + int(random() * 1000)
            ]

            for i in range(len(sensors)):
                p.produce(f"{topic}_{i}", json.dumps({"value": sensors[i]}),
                          timestamp=int((datetime.now().timestamp() - 3600) * 1000),
                          key=f"sensor_{i}",
                          callback=delivery_callback)
        except BufferError as e:
            sys.stderr.write('%% Local producer queue is full (%d messages awaiting delivery): try again\n' %
                             len(p))
        p.poll(0)

        time.sleep(1)

    sys.stderr.write('%% Waiting for %d deliveries\n' % len(p))
    p.flush()