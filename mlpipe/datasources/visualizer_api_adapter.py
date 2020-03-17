import datetime
import urllib
from typing import List

import pandas as pd
import requests
from requests_ntlm import HttpNtlmAuth

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
        for f in _fields:
            self.logger.info(f"getting data for field: {f.name} aka. {f.alias}")
            url = self._get_url(sensor_id=f.name, date_from=self.date_from, date_to=self.date_to)
            self.logger.debug(f"download data from: {url}")
            resp = self.session.get(url)

            if resp.status_code != 200:
                raise Exception(resp.text)

            if resp.text.strip() == "[]":
                raise Exception(f"no data found for sensor {f.name} in given time period")

            serie = pd.read_json(resp.text)
            serie = serie.set_index('timestamp')
            serie.index = serie.index.round("T")
            serie = serie.rename({"value": f.name}, axis=1)
            series.append(serie)

        df = pd.concat(series, axis=1)
        return StandardDataFormat.from_dataframe(df)

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
