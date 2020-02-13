import unittest
from dataclasses import dataclass
import json
from typing_extensions import TypedDict


@dataclass
class ExampleDataClass:
    name: str
    age: int


class ExampleTypedDict(TypedDict):
    name: str
    age: int


class MyTestCase(unittest.TestCase):
    def test_typed_dict(self):
        a = ExampleTypedDict(name="Bob", age=31)
        self.assertTrue(a['name'], "Bob")

        s = json.dumps(a)
        self.assertEqual('{"name": "Bob", "age": 31}', s)

    def test_typed_dict_unmarshall(self):
        desc = '{ "name": "Bob", "age": 34 }'
        a = json.loads(desc)
        self.assertTrue(a['name'], "Bob")


if __name__ == '__main__':
    unittest.main()
