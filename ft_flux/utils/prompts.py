from importlib import resources
import os
import functools
import random
import inflect
import json
IE = inflect.engine()
ASSETS_PATH = resources.files("utils.assets")


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `d3po_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}


def imagenet_all():
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    return from_file("simple_animals.txt")

def simple_animals_test():
    return from_file("simple_animals_test.txt")

def aesthetic():
    return from_file("aesthetic.txt")

def sci():
    path = os.path.join("utils/assets", 'reality_simple_300.json')
    data = json.load(open(path, 'r'))
    data = random.choice(data)
    return data['prompt'], data['object'], {}


def sci_eval():
    path = os.path.join("utils/assets", 'reality_hard_test.json')
    data = json.load(open(path, 'r'))
    data = random.choice(data)
    return data['prompt'], data['object'], {}