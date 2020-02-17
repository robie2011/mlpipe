import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import sleep
from typing import List
import requests
from ordered_set import OrderedSet
from requests_ntlm import HttpNtlmAuth
from typing_extensions import TypedDict
from mlpipe.config.app_settings import AppConfig
from mlpipe.datasources.visualizer_api_adapter import VisualizerApiAdapter
from mlpipe.integration.output.internal.csv_stream_writer import CsvStreamWriter
from mlpipe.utils.path_tool import dir_code
import sys

module_logger = logging.getLogger(__file__)

username = AppConfig['unit_test.visualizer_api_auth.username']
password = AppConfig['unit_test.visualizer_api_auth.password']


metric = sys.argv[1]
frequency_seconds = 30
output_path = dir_code / "output" / f"{metric}.csv"

module_logger.info(f"starting live data monitoring for metric = {metric}")
module_logger.info(f"choosen frequency (seconds) = {frequency_seconds}")
module_logger.info(f"output path is = {output_path}")

is_file_exists = output_path.is_file()


session = requests.Session()
session.auth = HttpNtlmAuth(username=username, password=password)


class TimelineEntryDict(TypedDict):
    value: float
    timestamp: str


@dataclass
class InternalCache:
    nitems: int = 50
    data: OrderedSet = OrderedSet()

    def add(self, s: str):
        if s not in self.data:
            self.data.append(s)
            if len(self.data) > self.nitems:
                self.data.remove(self.data.items[0])
            return True
        else:
            return False


def download() -> List[TimelineEntryDict]:
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

    return resp.json()


def process():
    cache = InternalCache()
    with CsvStreamWriter(headers=["timestamp", "value"], path=output_path) as writer:
        while True:
            updates = [a for a in download() if cache.add(a['timestamp'])]
            if updates:
                module_logger.info(f"writing updates ({len(updates)})")

            for b in updates:
                writer.write([b['timestamp'], b['value']])

            sleep(frequency_seconds)


if __name__ == '__main__':
    process()
