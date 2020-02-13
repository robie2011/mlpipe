from mlpipe.admin.package_class_finder import create_imports
from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.datasources.abstract_datasource_adapter import AbstractDatasourceAdapter
from mlpipe.groupers.abstract_grouper import AbstractGrouper
from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from mlpipe.integration.output.interface import AbstractOutput
from os import path

root_dir = path.dirname(path.dirname(path.abspath(path.abspath(__file__))))

tasks = (
    ("mlpipe.processors", AbstractProcessor, "processors"),
    ("mlpipe.datasources", AbstractDatasourceAdapter, "sources"),
    ("mlpipe.groupers", AbstractGrouper, "groupers"),
    ("mlpipe.aggregators", AbstractAggregator, "aggregators"),
    ("mlpipe.integration.output", AbstractOutput, "sinks"),
)

for package, base, group in tasks:
    print(f"create file for {package} / {base.__name__}")
    clazz = create_imports(package, base)
    output_file = path.join(root_dir, "dsl", f"{group}.py")
    lines = [f"from {c[0]} import {c[1]}" for c in clazz]
    with open(output_file, "w") as f:
        for i in lines:
            f.write("# noinspection PyUnresolvedReferences\r\n")
            f.write(i)
            f.write("\r\n")

