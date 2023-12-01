import os
import pkgutil
import importlib


filename_without_extension = os.path.splitext(os.path.basename(__file__))[0]

# Define a global set to store the package names
_packages_to_reload = set()
_modules_to_reload = set()

# ipython
ip = None


def reload_all_modules(package_name):
    """Reloads all submodules and subpackages under a specified package.
    
    This function is designed to work within environments like Jupyter notebooks 
    where modules may need to be frequently reloaded after changes. It takes a
    package name as an argument and reloads all of its submodules and subpackages.

    Args:
        package_name (str): The name of the package to reload its submodules and subpackages.

    Usage:
        reload_all_modules('ttsxai')

    Note:
        Reloading a module doesn't reset any dynamically generated states within that module.
    """
    global _package_to_reload

    package = importlib.import_module(package_name)

    # Iterates through all the submodules and subpackages within the specified package
    # and reloads each one of them.
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package_name + '.'):
        if module_name.split('.')[-1] != filename_without_extension:  # make sure we don't reload this file
            module = importlib.import_module(module_name)
            importlib.reload(module)


def reload_specific_modules(*module_names):
    """Reloads the specified modules."""
    for module_name in module_names:
        module = importlib.import_module(module_name)
        importlib.reload(module)


def always_reload(info):
    """Reload registered modules and packages before executing a cell."""
    global _packages_to_reload, _modules_to_reload
    
    for package in _packages_to_reload:
        reload_all_modules(package)
        
    for module in _modules_to_reload:
        reload_specific_modules(module)

    # Explicitly re-import the modules
    for module in _modules_to_reload:
        exec(f'from {module} import *', ip.user_ns)
    
    # Explicitly re-import the packages and all its submodules/subpackages
    for package in _packages_to_reload:
        package_module = importlib.import_module(package)
        for _, module_name, _ in pkgutil.walk_packages(package_module.__path__, package + '.'):
            exec(f'from {module_name} import *', ip.user_ns)

    # Set matplotlib to inline mode for Jupyter Notebook
    ip.run_line_magic('matplotlib', 'inline')


def register_always_reload(packages=[], modules=[]):
    """
    Register packages/modules to be reloaded before executing any cell.
    
    Args:
        packages (list): List of package names to be reloaded.
        modules (list): List of module names to be reloaded.
        
    Usage:
        always_reload.register_always_reload(packages=['ttsxai'], modules=['ttsxai.utils.plot_utils'])
    """
    global _packages_to_reload, _modules_to_reload, ip
    _packages_to_reload.update(packages)
    _modules_to_reload.update(modules)
    ip = get_ipython()
    get_ipython().events.register('pre_run_cell', always_reload)


def unregister_always_reload():
    """
    Unregister the always_reload function from pre_run_cell event and clear the lists.
    """
    global _packages_to_reload, _modules_to_reload
    _packages_to_reload.clear()
    _modules_to_reload.clear()
    get_ipython().events.unregister('pre_run_cell', always_reload)