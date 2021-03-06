import argparse
import logging

# fixing: backend info message from tensorflow by redirecting stdout
# https://stackoverflow.com/questions/51456842/how-do-i-stop-keras-showing-using-xxx-backend
import os, sys
from mlpipe.config.app_config_parser import ENVIRONMENT_VAR_PREFIX

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from mlpipe.cli.actions import list_models, describe_model, train_model, evaluate_model, analyze_data, integrate_model, \
    export_data
from mlpipe.exceptions.interface import MLException

sys.stderr = stderr

module_logger = logging.getLogger(__name__)


def action_not_implemented(args):
    print("No subcommand entered. Use --help to see available commands.", args.action)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    # train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument("files", metavar="FILES", default=[], nargs='*')

    # test
    parser_evaluate = subparsers.add_parser('evaluate')
    parser_evaluate.add_argument("files", metavar="FILES", default=[], nargs='*')

    # integrate
    parser_integrate = subparsers.add_parser('integrate')
    parser_integrate.add_argument("file", metavar="FILE")

    # list
    subparsers.add_parser('list')

    # describe
    parser_describe = subparsers.add_parser("describe")
    parser_describe.add_argument("model_session", metavar="MODEL/SESSION_ID")

    # analyze
    parser_analyze = subparsers.add_parser("analyze")
    parser_analyze.add_argument("--json", "-j", action='store_true')
    parser_analyze.add_argument("files", metavar="FILES", default=[], nargs='*')

    parser_export = subparsers.add_parser("export")
    parser_export.add_argument("files", metavar="FILES", default=[], nargs='*')
    parser_export.add_argument("--pipelinePrimary", '-p', action='store_true')
    parser_export.add_argument("--full", '-f', action='store_true')

    args = parser.parse_args()
    action_switcher = {
        "train": train_model,
        "list": list_models,
        "describe": describe_model,
        "evaluate": evaluate_model,
        "analyze": analyze_data,
        "integrate": integrate_model,
        "export": export_data
    }

    domain_exception_classes = tuple([MLException] + object.__class__.__subclasses__(MLException))

    try:
        for item, value in os.environ.items():
            if item.startswith(ENVIRONMENT_VAR_PREFIX):
                module_logger.info(f"Environment variable set: {item}={value}")

        action_switcher.get(args.action, action_not_implemented)(args)
    except domain_exception_classes as e:
        module_logger.error(e)
        exit(1)


if __name__ == "__main__":
    main()

