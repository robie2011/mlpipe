import datetime
import urllib
from typing import List
import pandas as pd
import requests
from requests_ntlm import HttpNtlmAuth
from mlpipe.datasources.abstract_datasource_adapter import AbstractDatasourceAdapter, Field
from mlpipe.processors.standard_data_format import StandardDataFormat


class VisualizerApiAdapter(AbstractDatasourceAdapter):
    def __init__(self,
                 username: str,
                 password: str,
                 date_from: str,
                 date_to: str,
                 fields: List[str] = [],
                 data_collection: str = None):
        if not data_collection and fields:
            raise ValueError("Property required: fields or data_collection")
        super().__init__(fields=fields)

        self.session = requests.Session()
        self.session.auth = HttpNtlmAuth(username=username, password=password)
        self.date_from = datetime.datetime.fromisoformat(date_from)
        self.date_to = datetime.datetime.fromisoformat(date_to)

        if data_collection:
            for name in self._get_collection_points(session=self.session, name=data_collection):
                if name in self.fields:
                    continue
                self.logger.info(f"adding field from datapoint collection: {name}")
                self.fields.append(name)

    def _fetch(self, _fields: List[Field]) -> StandardDataFormat:
        delta = (self.date_to - self.date_from)
        self.logger.info(
            f"Requested timewindow for download is from {self.date_from} to {self.date_to}. Window size is: {delta}")
        series = []
        for f in _fields:
            self.logger.info(f"getting data for field: {f.name} aka. {f.alias}")
            url = self._get_url(sensor_id=f.name, date_from=self.date_from, date_to=self.date_to)
            self.logger.debug(f"download data from: {url}")
            resp = self.session.get(url)
            if resp.status_code != 200:
                raise Exception(resp.text)

            if resp.text.strip() == "[]":
                msg = f"No measurements found for sensor: {f.name} in given time period: {self.date_from} to {self.date_to}."
                raise Exception(msg)

            serie = pd.read_json(resp.text)
            serie = serie.set_index('timestamp')
            serie = serie.rename({"value": f.name}, axis=1)
            series.append(serie)

        df = pd.concat(series, axis=1)
        return StandardDataFormat(
            timestamps=df.index.values,
            data=df.values,
            labels=df.columns.values.tolist()
        )

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

    @staticmethod
    def _get_collection_points(session: requests.session, name: str):
        url = f"https://visualizer.nestcollaboration.ch/Backend/api/v1/datapointcollections/{name}"
        resp = session.get(url)
        return resp.json()['items']
