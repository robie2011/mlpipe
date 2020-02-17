from dataclasses import dataclass
from time import sleep
from typing import List
import pandas as pd
import requests
from ordered_set import OrderedSet
from requests_ntlm import HttpNtlmAuth
from typing_extensions import TypedDict
from mlpipe.config.app_settings import AppConfig
from datetime import datetime, timedelta
from mlpipe.datasources.visualizer_api_adapter import VisualizerApiAdapter
import dateutil.parser as dparser


username = AppConfig['unit_test.visualizer_api_auth.username']
password = AppConfig['unit_test.visualizer_api_auth.password']


metric = "40210012"
session = requests.Session()
session.auth = HttpNtlmAuth(username=username, password=password)


class TimelineEntryDict(TypedDict):
    value: float
    timestamp: str


@dataclass
class TimelineEntry:
    value: float
    timestamp: datetime

    def __post_init__(self):
        self._stamp = int(self.timestamp.timestamp())

    def __hash__(self):
        return self._stamp


def download() -> List[TimelineEntry]:
    date_to = datetime.now()
    date_from = date_to - timedelta(minutes=3)
    # noinspection PyProtectedMember
    url = VisualizerApiAdapter._get_url(
        sensor_id=metric,
        date_from=date_from,
        date_to=date_to
    )
    resp = session.get(url)

    if resp.status_code != 200:
        raise Exception(resp.text)

    if resp.text.strip() == "[]":
        raise Exception(f"no data found for sensor {metric} in given time period")

    results: List[TimelineEntryDict] = resp.json()
    return list(map(lambda x: TimelineEntry(
        value=x['value'], timestamp=dparser.isoparse(x['timestamp'])), results))


def process():
    cache = OrderedSet()

    while True:
        need_update = False
        for b in [a for a in download() if a not in cache]:
            cache.add(b)
            need_update = True

        if need_update:
            data_object = {
                "timestamp": [],
                "value": []
            }

            for c in cache:
                data_object['timestamp'].append(c.timestamp)
                data_object['value'].append(c.value)

            df = pd.DataFrame(data_object)
            df = df.set_index('timestamp')
            df.index = df.index.round("T")
            df = df.rename({"value": metric}, axis=1)
            print(df)

        sleep(10)

process()
