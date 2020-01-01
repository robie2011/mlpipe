import json
import os
from pathlib import Path
from typing import Dict
import tabulate
from mlpipe.config import app_settings
import logging


module_logger = logging.getLogger(__name__)


class AnalyticsDataManager:
    @staticmethod
    def list_files():
        path = Path(app_settings.dir_analytics)
        analytics_files = list(path.glob("*.json"))
        print(f"Saved analytics descriptions {app_settings.dir_analytics} ({len(analytics_files)} Files):")
        if len(analytics_files) == 0:
            module_logger.info("Directory is empty.")
        else:
            results = []
            for analytics_file in analytics_files:
                basename = os.path.basename(analytics_file)
                name, ext = os.path.splitext(basename)
                results.append([name, analytics_file])
            print("")
            print(tabulate.tabulate(results, headers=['name', 'path']))

    @staticmethod
    def save(name: str, description: Dict, overwrite=False):
        path = os.path.join(app_settings.dir_analytics, name + ".json")
        if not os.path.isfile(path) or overwrite:
            with open(path, "w") as f:
                json.dump(description, f)
                module_logger.info(f"analytics description created: {name}")
        else:
            print(f"Can not save {name}. This name already exists ({path}). Choose another name.")

    @staticmethod
    def get(name) -> Dict:
        path = os.path.join(app_settings.dir_analytics, name + ".json")
        if not os.path.isfile(path):
            raise Exception(f"Invalid analytics description name: {name} (path)")
        else:
            with open(path, "w") as f:
                return json.load(f)

    @staticmethod
    def delete(name):
        path = os.path.join(app_settings.dir_analytics, name + ".json")
        if not os.path.isfile(path):
            raise Exception(f"Invalid analytics description name: {name} ({path})")
        else:
            os.remove(path)
