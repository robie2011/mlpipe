import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import sys
import os
from confluent_kafka import Producer

topic = "test02"
topic_prefix = "signal_"
nrows = None
#topic = os.environ['CLOUDKARAFKA_TOPIC']

# Consumer configuration
# See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
conf = {
    'bootstrap.servers': os.environ['CLOUDKARAFKA_BROKERS'],
    'session.timeout.ms': 6000,
    'default.topic.config': {'auto.offset.reset': 'smallest'},
    'queue.buffering.max.messages': 1000000
    # 'security.protocol': 'SASL_SSL',
    # 'sasl.mechanisms': 'SCRAM-SHA-256',
    # 'sasl.username': os.environ['CLOUDKARAFKA_USERNAME'],
    # 'sasl.password': os.environ['CLOUDKARAFKA_PASSWORD']
}

p = Producer(**conf)

def delivery_callback(err, msg):
    if err:
        sys.stderr.write('%% Message failed delivery: %s\n' % err)



def load(path:str):
    cwd = os.path.dirname(__file__)
    out = pd.read_csv(
        path,
        sep=',',
        date_parser=lambda x: datetime.strptime(x, '%d.%m.%Y %H:%M:%S'),
        parse_dates=['_TIMESTAMP'], nrows=nrows)

    # cleaning
    out = out.rename(columns={'_TIMESTAMP': 'TIMESTAMP'})
    out = out.dropna()

    # index number has changed because nans are dropped
    out.reset_index(inplace=True)

    # remove old index column
    out = out.drop(labels='index', axis=1)
    return out


df = load("/Users/robert.rajakone/repos/2019_p8/code/trainframework/meeting_room_sensors_201807_201907.csv")
df['unixtime'] = df['TIMESTAMP'].values.astype('datetime64[ms]').astype('int')

i = 0
for col_id in range(1, df.shape[1]-1):
    print(">>> col_id", col_id, df.columns[col_id])
    for (stamp, measurement_value) in df.iloc[:, [-1, col_id]].values:
        try:
            topic = topic_prefix + df.columns[col_id]
            p.produce(topic, str(measurement_value), timestamp=int(stamp), callback=delivery_callback)
        except BufferError as e:
            sys.stderr.write('%% Local producer queue is full (%d messages awaiting delivery): try again\n' %
                             len(p))
        p.poll(0)

        i += 1
        if i % 10000 == 0:
            print("count", i)
            p.flush()

sys.stderr.write('%% Waiting for %d deliveries\n' % len(p))
p.flush()
