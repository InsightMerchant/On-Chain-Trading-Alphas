import importlib
import os
from utils.interfaces import LogicInterface

class PluginLoader:
    def __init__(self, logic_package="strategies.logic"):
        self.logic_package = logic_package

    def load_logic_plugins(self):
        plugins = {}
        # Build an absolute path using the current working directory
        package_path = os.path.join(os.getcwd(), *self.logic_package.split('.'))
        if not os.path.exists(package_path):
            print(f"Logic package directory not found: {package_path}")
            return plugins
        for filename in os.listdir(package_path):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]
                full_module_name = f"{self.logic_package}.{module_name}"
                try:
                    module = importlib.import_module(full_module_name)
                    for attr in dir(module):
                        obj = getattr(module, attr)
                        if (isinstance(obj, type) and 
                            issubclass(obj, LogicInterface) and 
                            obj is not LogicInterface):
                            plugins[module_name] = obj()
                except Exception as e:
                    print(f"Error loading logic plugin '{module_name}': {e}")
        return plugins
