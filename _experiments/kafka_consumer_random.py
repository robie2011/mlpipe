import sys
import os
import json
import time

from confluent_kafka import Consumer, KafkaException, KafkaError

if __name__ == '__main__':
    #topics = os.environ['CLOUDKARAFKA_TOPIC'].split(",")
    topics = ['7msxg02r-random1']

    # Consumer configuration
    # See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
    conf = {
        'bootstrap.servers': os.environ['CLOUDKARAFKA_BROKERS'],
        'group.id': "random-consumers",
        'session.timeout.ms': 6000,
        'default.topic.config': {'auto.offset.reset': 'smallest'},
    #     'security.protocol': 'SASL_SSL',
	# 'sasl.mechanisms': 'SCRAM-SHA-256',
    #     'sasl.username': os.environ['CLOUDKARAFKA_USERNAME'],
    #     'sasl.password': os.environ['CLOUDKARAFKA_PASSWORD']
    }

    c = Consumer(**conf)
    c.subscribe(topics)
    try:
        while True:
            msg = c.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                # Error or event
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                     (msg.topic(), msg.partition(), msg.offset()))
                elif msg.error():
                    # Error
                    raise KafkaException(msg.error())
            else:
                # Proper message
                sys.stderr.write('%% %s [%d] at offset %d with key %s:\n' %
                                 (msg.topic(), msg.partition(), msg.offset(),
                                  str(msg.key())))
                v = json.loads(msg.value())['value']
                print("received", v)
                #print("sleep for sec:", v)
                #time.sleep(v)


    except KeyboardInterrupt:
        sys.stderr.write('%% Aborted by user\n')

    # Close down consumer to commit final offsets.
    c.close()