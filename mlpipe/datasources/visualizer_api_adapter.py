import datetime
import urllib
from pathlib import Path
from typing import List

import pandas as pd
import requests
from requests_ntlm import HttpNtlmAuth
from diskcache import Cache

from mlpipe.config.app_settings import AppConfig
from mlpipe.datasources.abstract_datasource_adapter import AbstractDatasourceAdapter, Field
from mlpipe.processors.standard_data_format import StandardDataFormat


class VisualizerApiAdapter(AbstractDatasourceAdapter):
    def __init__(self, fields: List[str],
                 username: str,
                 password: str,
                 date_from: str,
                 date_to: str):
        super().__init__(fields=fields)
        self.session = requests.Session()
        self.session.auth = HttpNtlmAuth(username=username, password=password)

        # workaround: dateutil.parser.isoparse
        self.date_from = datetime.datetime.fromisoformat(date_from)
        self.date_to = datetime.datetime.fromisoformat(date_to)

    def _fetch(self, _fields: List[Field]) -> StandardDataFormat:
        self.logger.info(f"getting data from time period {self.date_from.isoformat()} - {self.date_to.isoformat()}")
        self.logger.info(f"using username for authentification {self.session.auth.username}")
        series = []
        for field in _fields:
            self.logger.info(f"getting data for field: {field.name} aka. {field.alias}")
            url = self._get_url(sensor_id=field.name, date_from=self.date_from, date_to=self.date_to)
            self.logger.debug(f"download data from: {url}")
            serie = pd.read_json(self._get_data(url, field))
            serie = serie.set_index('timestamp')
            serie.index = serie.index.round("T")
            serie = serie.rename({"value": field.name}, axis=1)
            dups = serie.index.duplicated()
            if len(dups):
                self.logger.info(f"dropping {len(dups)} duplicated (index) measurements from {field}")
                serie = serie[~dups]
            series.append(serie)
            self.logger.debug(f"received rows: {len(serie)}")

        df = pd.concat(series, axis=1)
        return StandardDataFormat.from_dataframe(df)

    def _get_data(self, url, field_name):
        with Cache(Path(AppConfig['general.dir_cache']) / "diskcache") as cache:
            if url not in cache:
                self.logger.debug("Url Response not in http cache. Downloading.")
                try:
                    resp = self.session.get(url)
                except requests.exceptions.ConnectionError as e:
                    self.logger.error(f"can not connect to server")
                    raise

                if resp.status_code != 200:
                    if resp.status_code == 401:
                        self.logger.error(f"Authetification error: Check username/password configuration")
                    else:
                        self.logger.error(f"Received invalid response from server. Code {resp.status_code}")
                    raise Exception(resp.text)

                if resp.text.strip() == "[]":
                    raise Exception(f"no data found for sensor {field_name.name} in given time period")

                cache[url] = resp.text
            return cache[url]


    @staticmethod
    def _get_url(sensor_id: str, date_from: datetime.date, date_to: datetime.datetime) -> str:
        params = {
            "startDate": date_from.isoformat(),
            "endDate": date_to.isoformat()
        }

        url = "".join([
            "https://visualizer.nestcollaboration.ch/Backend/api/v1/datapoints/",
            sensor_id,
            "/timeline?"
        ])

        return url + urllib.parse.urlencode(params)
