"""kanboost.registry -- pushing/pulling a saved model to a remote
object store (`mlhub`). `LocalRegistry` (a local, versioned model
registry) lands in a later restructure PR."""

from .mlhub import push_model, pull_model, list_models, ensure_bucket
