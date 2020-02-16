import unittest

from mlpipe.config.app_config_parser import AppConfigParser, MissingConfigurationException


class TestAppConfig(unittest.TestCase):
    def setUp(self) -> None:
        config_dict = {
            "name": "Robert",
            "address": {
                "country": "Switzerland",
                "city": "Schaffhausen"
            },
            "dir_tmp": "/nonexisting/example/folder1",
            "training": {
                "dir_data": "/nonexisting/example/folder2"
            }
        }
        self.config = AppConfigParser(config=config_dict, autocreate_dirs=False)

    def test_get_config_basic(self):
        self.assertEqual("Robert", self.config.get_config("name"))
        self.assertEqual("Switzerland", self.config.get_config("address.country"))
        self.assertEqual("Robert", self.config["name"])

    def test_get_config(self):
        self.assertRaises(MissingConfigurationException,
                          lambda: self.config.get_config("address.country.code"))

    def test_get_config_or_default(self):
        self.assertEqual("CH",
                         self.config.get_config_or_default("name.address.country.code", "CH"))

    def test__find_dir_keys(self):
        names = self.config._find_dir_keys(self.config.config)
        self.assertIn(['dir_tmp', '/nonexisting/example/folder1'], names)
        self.assertIn(['dir_data', '/nonexisting/example/folder2'], names)

    def test___post_init(self):
        import pathlib
        import tempfile
        import os
        tmp_path = pathlib.Path(tempfile.gettempdir()) / "mlpipe_unit_test_create_dir_for_app_settings"
        if tmp_path.is_dir():
            os.rmdir(tmp_path)

        config_dict = {
            "training": {
                "dir_data": str(tmp_path)
            }
        }
        config = AppConfigParser(config_dict)
        self.assertTrue(pathlib.Path(config['training.dir_data']).is_dir())

        if tmp_path.is_dir():
            os.rmdir(tmp_path)
