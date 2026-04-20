import numpy as np
import os
import collections
import collections.abc
import random
import re
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def _slug(s):
    # Keep run-folder names filesystem-safe and predictable.
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s)).strip('_') or "x"


def _extract_with_override(params, key):
    # Sacred "with k=v" overrides take priority over config-file values.
    prefix = key + "="
    for p in params:
        if p.startswith(prefix):
            return p[len(prefix):]
    return None


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env", "envs")
    alg_config = _get_config(params, "--alg", "algs")
    map_config = _get_config(params, "--map", "maps")
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    config_dict = recursive_dict_update(config_dict, map_config)

    # Fix the seed up-front so the run folder name is deterministic and known
    # before Sacred starts. CLI "with seed=N" still wins.
    seed_override = _extract_with_override(params, "seed")
    if seed_override is not None:
        config_dict["seed"] = int(seed_override)
    elif config_dict.get("seed") in (None, 0):
        config_dict["seed"] = random.randint(0, 2**31 - 1)

    # Allow env_args.map_name override from CLI for the folder name.
    map_override = _extract_with_override(params, "env_args.map_name")
    if map_override is not None:
        config_dict.setdefault("env_args", {})["map_name"] = map_override

    alg_name = config_dict.get("name", "alg")
    env_name = config_dict.get("env", "env")
    map_name = config_dict.get("env_args", {}).get("map_name", "map")
    seed_val = config_dict["seed"]

    run_name = "{}_seed{}_{}_{}".format(
        _slug(alg_name), seed_val, _slug(env_name), _slug(map_name)
    )
    run_dir = os.path.join(results_path, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Expose the deterministic run directory to the rest of the pipeline.
    config_dict["local_results_path"] = run_dir
    config_dict["run_name"] = run_name

    # Force Sacred to use the exact seed we baked into the folder name,
    # rather than generating its own at run start. Sacred's CLI requires
    # "with key=value" tokens; splice "with" in if it's not already there.
    if seed_override is None:
        if "with" not in params:
            params.append("with")
        params.append("seed={}".format(seed_val))

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Each run gets its own sacred directory under the run folder.
    sacred_dir = os.path.join(run_dir, "sacred")
    logger.info("Saving to FileStorageObserver in {}".format(sacred_dir))
    ex.observers.append(FileStorageObserver.create(sacred_dir))

    ex.run_commandline(params)
