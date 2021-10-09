import json
from collections import Mapping
from copy import deepcopy


class ParamDict(dict):
    """
    An immutable dict where elements can be accessed with a dot.
    """

    def __getattr__(self, *args, **kwargs):
        try:
            return self.__getitem__(*args, **kwargs)
        except KeyError as e:
            raise AttributeError(e)

    def __delattr__(self, item):
        raise TypeError("Setting object not mutable after settings are fixed!")

    def __setattr__(self, key, value):
        raise TypeError("Setting object not mutable after settings are fixed!")

    def __setitem__(self, key, value):
        raise TypeError("Setting object not mutable after settings are fixed!")

    def __deepcopy__(self, memo):
        """ In order to support deepcopy"""
        return ParamDict([(deepcopy(k, memo), deepcopy(v, memo)) for k, v in self.items()])

    def __repr__(self):
        return json.dumps(self, indent=4, sort_keys=True)


def recursive_objectify(nested_dict):
    """
    Turns a nested_dict into a nested ParamDict.
    """
    result = deepcopy(nested_dict)
    for k, v in result.items():
        if isinstance(v, Mapping):
            result[k] = recursive_objectify(v)
    return ParamDict(result)


def recursive_dictify(param_dict):
    """
    Turns a nested ParamDict into a nested_dict.
    """
    result = dict(deepcopy(param_dict))
    for k, v in result.items():
        if isinstance(v, Mapping):
            result[k] = recursive_dictify(v)
    return result
