from dataclasses import dataclass
from typing import Type, List

from mlpipe.admin.package_class_finder import create_imports
from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from mlpipe.datasources.abstract_datasource_adapter import AbstractDatasourceAdapter
from mlpipe.groupers.abstract_grouper import AbstractGrouper
from mlpipe.outputs.interface import AbstractOutput
from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.utils.path_tool import dir_mlpipe


@dataclass
class Task:
    package: str
    base: Type
    dsl_group: str
    exclude_types: List[Type] = ()


tasks = [
    Task(package="mlpipe.processors",
         base=AbstractProcessor,
         dsl_group="processors",
         exclude_types=[AbstractAggregator]),
    Task(package="mlpipe.datasources", base=AbstractDatasourceAdapter, dsl_group="sources"),
    Task(package="mlpipe.groupers", base=AbstractGrouper, dsl_group="groupers"),
    Task(package="mlpipe.aggregators", base=AbstractAggregator, dsl_group="aggregators"),
    Task(package="mlpipe.outputs", base=AbstractOutput, dsl_group="outputs"),
]

for task in tasks:
    print(f"create file for {task.package} / {task.base.__name__}")
    clazz = create_imports(task.package, task.base, exclude_types=task.exclude_types)
    output_file = dir_mlpipe / "dsl" / f"{task.dsl_group}.py"

    print(f"    found {len(clazz)} classes")
    lines = [f"from {c[0]} import {c[1]}" for c in clazz]
    with open(output_file, "w") as f:
        for i in lines:
            f.write("# noinspection PyUnresolvedReferences\r\n")
            f.write(i)
            f.write("\r\n")
