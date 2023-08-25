import inspect
import logging
from typing import Any, Optional, Set

from langsmith.run_helpers import traceable

logger = logging.getLogger(__name__)


def _get_module_path(module_class: type) -> str:
    """
    Returns the full module path of a given class.

    :param module_class: The class to get the module path for.
    :type module_class: type
    :return: The module path
    :rtype: str
    """
    return (
        getattr(module_class, "__module__", "")
        + "."
        + getattr(module_class, "__name__", "")
    ).strip(".")


class ModuleWrapper:
    __slots__ = ["_lc_module", "_lc_llm_paths", "__weakref__", "_run_type"]

    def __init__(self, module: Any, llm_paths: Optional[Set[str]] = None):
        object.__setattr__(self, "_lc_module", module)
        run_type = "chain"
        if llm_paths:
            full_path = _get_module_path(module)
            if full_path in llm_paths:
                run_type = "llm"
        object.__setattr__(self, "_run_type", run_type)
        object.__setattr__(self, "_lc_llm_paths", llm_paths or set())

    def __getattr__(self, name: str) -> Any:
        attr = getattr(object.__getattribute__(self, "_lc_module"), name)
        if inspect.isclass(attr) or inspect.isfunction(attr) or inspect.ismethod(attr):
            llm_paths = object.__getattribute__(self, "_lc_llm_paths")
            return self.__class__(attr, llm_paths=llm_paths)
        return attr

    def __setattr__(self, name, value):
        setattr(self._lc_module, name, value)

    def __delattr__(self, name):
        delattr(self._lc_module, name)

    def __call__(self, *args, **kwargs):
        function_object = object.__getattribute__(self, "_lc_module")
        if inspect.isclass(function_object):
            return self.__class__(
                function_object(*args, **kwargs),
            )
        run_name = _get_module_path(function_object)
        run_type = object.__getattribute__(
            self,
            "_run_type",
        )
        wrapped_func = traceable(run_type=run_type, name=run_name)(function_object)
        result = wrapped_func(*args, **kwargs)
        # If they share a common root domain
        this_root = _get_module_path(self._lc_module).split(".")[0]
        if _get_module_path(result).split(".")[0] == this_root:
            return self.__class__(
                result, llm_paths=object.__getattribute__(self, "_lc_llm_paths")
            )
        return result

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_lc_module"))
