from typing import Dict, List


def _get_descriptions_name(descriptions: List[Dict]) -> List[str]:
    return list(map(lambda x: x['name'], descriptions))
