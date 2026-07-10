"""kanboost.registry -- a local, versioned model registry
(`LocalRegistry`), and pushing/pulling a saved model to a remote object
store (`mlhub`)."""

from .mlhub import push_model, pull_model, list_models, ensure_bucket
from .local import LocalRegistry
