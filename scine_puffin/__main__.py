#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import argparse
import sys
from typing import Union
from .bootstrap import bootstrap
from .config import Configuration
from .daemon import start_daemon, stop_daemon, check_environment
from .jobloop import kill_daemon


def setup_config(args: argparse.Namespace) -> Configuration:
    config = Configuration()
    if args.config:
        config.load(args.config)
        print("Loading configuration: " + args.config)
        print("Applying environment variables afterwards.")
    else:
        print("No configuration loaded, using default configuration with environment variables.")
        config.load()
    return config


def parse_arguments(include_action: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCINE Puffin")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        metavar="path",
        help="The path to the configuration file.",
    )
    if include_action:
        parser.add_argument(
            "action",
            choices=["configure", "bootstrap", "start", "stop", "kill", "container"],
            help="The main action to be taken.",
        )
    return parser.parse_args()


def configure(args: Union[None, argparse.Namespace] = None):
    if args is None:
        args = parse_arguments()
    print("")
    print("+--------------------+")
    print("|  Configure Puffin  |")
    print("+--------------------+")
    config = Configuration()
    print("Applying environment variables to default configuration, then dumping into a .yaml file.")
    if not args.config:
        args.config = "puffin.yaml"
    config.dump(args.config)
    print("")
    print("Please edit the generated '" + args.config + "' to your liking.")


def main_bootstrap(config: Union[None, Configuration] = None):
    print("")
    print("+-----------------+")
    print("|  Bootstrapping  |")
    print("+-----------------+")
    print("")
    print("Getting Puffin ready for use.")
    print("")
    if config is None:
        config = setup_config(parse_arguments())
    bootstrap(config)


def stop(config: Union[None, Configuration] = None):
    print("")
    print("+-------------------+")
    print("|  Stopping Puffin  |")
    print("+-------------------+")
    print("")
    print("Puffin will stop after the current job has concluded.")
    if config is None:
        config = setup_config(parse_arguments())
    stop_daemon(config)
    print("")
    print("Goodbye!")


def kill(config: Union[None, Configuration] = None):
    print("")
    print("+------------------+")
    print("|  Killing Puffin  |")
    print("+------------------+")
    if config is None:
        config = setup_config(parse_arguments())
    kill_daemon(config)
    print("")
    print("Goodbye!")


def start(config: Union[None, Configuration] = None):
    print("")
    print("+-------------------+")
    print("|  Starting Puffin  |")
    print("+-------------------+")
    if config is None:
        config = setup_config(parse_arguments())
    print("")
    print("Running with UUID: " + config["daemon"]["uuid"])
    start_daemon(config)


def container(config: Union[None, Configuration] = None):
    print("")
    print("+----------------------------------+")
    print("|  Starting Puffin in Docker Mode  |")
    print("+----------------------------------+")
    if config is None:
        config = setup_config(parse_arguments())
    print("")
    print("Running with UUID: " + config["daemon"]["uuid"])
    check_environment(config)
    start_daemon(config, detach=False)


def main():
    args = parse_arguments(include_action=True)
    if args.action == "configure":
        configure(args)
        sys.exit()
    config = setup_config(args)
    if args.action == "bootstrap":
        main_bootstrap(config)
    elif args.action == "stop":
        stop(config)
    elif args.action == "kill":
        kill(config)
    elif args.action == "start":
        start(config)
    elif args.action == "container":
        container(config)
    else:
        raise NotImplementedError("Action " + str(args.action) + " has not been implemented properly")


if __name__ == "__main__":
    main()
