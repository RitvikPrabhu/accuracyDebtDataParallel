import importlib
import logging

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

quantom_log = logging.getLogger(__name__)


def get_module(name, module_type, class_map):
    if name in class_map:
        module_name = class_map[name]
        module = __import__(f"{module_type}.{module_name}", fromlist=[name])
        return getattr(module, name)
    else:
        raise AttributeError(f"module '{module_type}' has no attribute '{name}'")


def load(name):
    mod_name, attr_name = name.split(":")
    quantom_log.debug(f"Attempting to load {mod_name} with {attr_name}")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class ModuleSpec(object):
    def __init__(self, id, entry_point, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the desired module with appropriate kwargs"""
        if self.entry_point is None:
            error_msg = (
                f"Attempting to make deprecated module {self.id}. "
                "(HINT: is there a newer registered version of this agent?)"
            )
            quantom_log.error(error_msg)
            raise ValueError(error_msg)

        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            module = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            module = cls(**_kwargs)

        return module


class ModuleRegistry(object):
    def __init__(self):
        self.module_specs = {}

    def make(self, path, **kwargs):
        #        if len(kwargs) > 0:
        #            quantom_log.debug(f"Making new module: {path} -- {kwargs}")
        #        else:
        quantom_log.info(f"Making new module: {path}")
        specs = self.spec(path)
        module = specs.make(**kwargs)

        return module

    def all(self):
        return self.module_specs.values()

    def spec(self, path):
        if ":" in path:
            mod_name, _sep, id = path.partition(":")
            try:
                importlib.import_module(mod_name)
            except ImportError as e:
                error_msg = (
                    f"A module ({mod_name}) was specified but was not found, "
                    "make sure the package is installed with `pip install` before "
                    "calling `quantom.make()`'."
                )
                quantom_log.error(error_msg)
                raise ImportError(error_msg) from e

        else:
            id = path
        try:
            return self.module_specs[id]
        except KeyError as e:
            error_msg = f"No registered module with id: {id}"
            quantom_log.error(error_msg)
            raise KeyError(error_msg) from e

    def register(self, id, **kwargs):
        if id in self.module_specs:
            error_msg = f"Cannot re-register id: {id}"
            quantom_log.error(error_msg)
            raise KeyError(error_msg)
        quantom_log.debug(f"Registering {id} with kwargs: {kwargs}")
        self.module_specs[id] = ModuleSpec(id, **kwargs)


# Global module registry
quantom_registry = ModuleRegistry()


def register(id, **kwargs):
    return quantom_registry.register(id, **kwargs)


def make(id, **kwargs):
    return quantom_registry.make(id, **kwargs)


def spec(id):
    return quantom_registry.spec(id)


def list_registered_modules():
    return list(quantom_registry.module_specs.keys())


def register_class(class_obj):
    register(id=class_obj.__name__, entry_point=class_obj)
    return class_obj


class GroupStore:
    def __init__(self, group_name: str):
        self.group_name = group_name
        self.cs = ConfigStore.instance()

    def store(self, name, node):
        self.cs.store(group=self.group_name, name=name, node=node)


def register_with_hydra(group, defaults, name):
    cs = ConfigStore.instance()
    if isinstance(group, str):
        group = [group]

    def decorator(class_obj):
        if class_obj.__name__ != defaults.id:
            raise ValueError(
                "Configuration and class name mismatch for "
                f"{name} in {group}. "
                f"{class_obj.__name__} != {defaults.id}"
            )
        register_class(class_obj)
        for g in group:
            cs.store(group=g, name=name, node=defaults)
        return class_obj

    return decorator
