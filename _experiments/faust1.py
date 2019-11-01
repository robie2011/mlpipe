import faust
import json
from faust.models import Record

app = faust.App(
    'hello-world',
    broker='kafka://judasheet.com:9091',
    value_serializer='raw',

)


class RandomValueEvent(Record, serializer='json'):
    value: int


random_topic = app.topic('7msxg02r-random1', value_serializer=RandomValueEvent)
view_max_value = app.Table('random_max12345_12', default=list, partitions=4).hopping(10, 5)


@app.agent(random_topic)
async def printer(values):
    #async for event in values.events():
    async for event in values:
        #print((view_max_value[0].current(), view_max_value[0].now(), view_max_value[0].value()))
        # v_max = view_max_value[0].current()
        # v = event.value.value
        # if v_max < v:
        #     view_max_value[0] = v
        #     print("updated!")
        # print(f"event time max: {view_max_value[0].current()}. now max: {view_max_value[0].now()}")

        v = event.value
        lst = view_max_value[0].current()
        lst.append(v)
        view_max_value[0] = lst

        print(view_max_value[0].current(), view_max_value[0].now())

