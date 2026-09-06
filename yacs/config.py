import ast
import copy

import yaml


class CfgNode(dict):
    """Small yacs-compatible config node for this project.

    It implements the subset of yacs used by X3D: attribute access, cloning,
    freezing/defrosting, and merging from YAML files or command-line lists.
    """

    def __init__(self, init_dict=None):
        super().__init__()
        object.__setattr__(self, "_immutable", False)
        if init_dict:
            for key, value in init_dict.items():
                self[key] = self._wrap(value)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if self._immutable:
            raise AttributeError("Attempted to modify a frozen CfgNode")
        self[name] = self._wrap(value)

    def __setitem__(self, key, value):
        if self._immutable:
            raise AttributeError("Attempted to modify a frozen CfgNode")
        super().__setitem__(key, self._wrap(value))

    def clone(self):
        return copy.deepcopy(self)

    def freeze(self):
        self._set_immutable(True)

    def defrost(self):
        self._set_immutable(False)

    def merge_from_file(self, cfg_filename):
        with open(cfg_filename, "r") as handle:
            loaded = yaml.safe_load(handle) or {}
        self._merge_dict(loaded)

    def merge_from_list(self, cfg_list):
        if cfg_list is None:
            return
        if len(cfg_list) % 2 != 0:
            raise ValueError("Override list must contain KEY VALUE pairs")
        for key, value in zip(cfg_list[0::2], cfg_list[1::2]):
            self._set_by_path(key.split("."), self._decode(value))

    def _set_immutable(self, immutable):
        object.__setattr__(self, "_immutable", immutable)
        for value in self.values():
            if isinstance(value, CfgNode):
                value._set_immutable(immutable)

    def _merge_dict(self, values):
        for key, value in values.items():
            if isinstance(value, dict):
                if key not in self or not isinstance(self[key], CfgNode):
                    self[key] = CfgNode()
                self[key]._merge_dict(value)
            else:
                self[key] = self._decode(value)

    def _set_by_path(self, path, value):
        node = self
        for key in path[:-1]:
            if key not in node or not isinstance(node[key], CfgNode):
                node[key] = CfgNode()
            node = node[key]
        node[path[-1]] = value

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict) and not isinstance(value, CfgNode):
            return CfgNode(value)
        return value

    @classmethod
    def _decode(cls, value):
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                lowered = value.lower()
                if lowered == "true":
                    return True
                if lowered == "false":
                    return False
                return value
        if isinstance(value, list):
            return [cls._decode(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._decode(item) for key, item in value.items()}
        return value

    def __deepcopy__(self, memo):
        copied = CfgNode()
        memo[id(self)] = copied
        object.__setattr__(copied, "_immutable", self._immutable)
        for key, value in self.items():
            dict.__setitem__(copied, key, copy.deepcopy(value, memo))
        return copied

    def __str__(self):
        return yaml.safe_dump(self._to_plain_dict(), sort_keys=True)

    def _to_plain_dict(self):
        output = {}
        for key, value in self.items():
            if isinstance(value, CfgNode):
                output[key] = value._to_plain_dict()
            else:
                output[key] = value
        return output
