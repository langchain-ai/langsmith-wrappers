import langsmith

__version__ = getattr(langsmith, "__version__", "Unknown")
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
