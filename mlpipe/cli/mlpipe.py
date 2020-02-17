import argparse

from mlpipe.cli.actions import list_models, describe_model, train_model, test_model, analyze_data, integrate_model, \
    export_data


def action_not_implemented(args):
    print("Action not implemented", args.action)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    # train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument("files", metavar="FILES", default=[], nargs='*')

    # test
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument("files", metavar="FILES", default=[], nargs='*')

    # integrate
    parser_integrate = subparsers.add_parser('integrate')
    parser_integrate.add_argument("files", metavar="FILES", default=[], nargs='*')

    # list
    subparsers.add_parser('list')

    # describe
    parser_describe = subparsers.add_parser("describe")
    parser_describe.add_argument("model_session", metavar="MODEL/SESSION_ID")

    # analyze
    parser_analyze = subparsers.add_parser("analyze")
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
        "test": test_model,
        "analyze": analyze_data,
        "integrate": integrate_model,
        "export": export_data
    }

    action_switcher.get(args.action, action_not_implemented)(args)
