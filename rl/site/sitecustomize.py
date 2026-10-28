"""On PYTHONPATH in the rollout container (rl/rollout_inner.sh): every process that imports SGLang's server arguments
loads SGLang's general plugins right after, as SGLang's own engine processes do. miles checks the SGLang argv it
renders in Ray actors that never load plugins, so without this the Flint plugin's parser and grammar backend names
(flint) are refused there; miles' driver also reads the checkpoint's config, whose model type the plugin registers."""
import importlib.abc
import importlib.util
import sys


class _LoadPluginsAfterServerArgs(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "sglang.srt.server_args":
            return None
        sys.meta_path.remove(self)   # once, and not for the lookup below
        spec = importlib.util.find_spec(name)
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            from sglang.srt.plugins import load_plugins
            load_plugins()
        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _LoadPluginsAfterServerArgs())
