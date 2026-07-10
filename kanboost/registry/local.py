"""
kanboost.registry.local -- a thin local model registry: save, version,
and list fitted models by name, with an optional push to the existing
remote object store (kanboost.registry.mlhub) behind the same
interface. Not a rebuild of mlhub or mlflow_utils -- storage is just
`model.save()` files plus a small JSON manifest per name; no
staging/production aliases, no lineage graph, no artifact store.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


class LocalRegistry:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

    def _manifest_path(self, name: str) -> str:
        return os.path.join(self.root_dir, f"{name}_manifest.json")

    def _model_path(self, name: str, version: int) -> str:
        return os.path.join(self.root_dir, f"{name}_v{version}.pt")

    def _load_manifest(self, name: str) -> dict:
        path = self._manifest_path(name)
        if not os.path.exists(path):
            return {"name": name, "versions": []}
        with open(path) as f:
            return json.load(f)

    def _save_manifest(self, name: str, manifest: dict) -> None:
        with open(self._manifest_path(name), "w") as f:
            json.dump(manifest, f, indent=2)

    def register(self, model, name: str, tags: dict | None = None) -> int:
        """Save `model` under `name`, returning the new version number
        (1, 2, 3, ... per name, independent of other names)."""
        manifest = self._load_manifest(name)
        version = len(manifest["versions"]) + 1
        model.save(self._model_path(name, version))
        manifest["versions"].append({
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": model.get_params(),
            "tags": tags or {},
        })
        self._save_manifest(name, manifest)
        return version

    def get(self, name: str, version="latest"):
        """Load a registered model. `version="latest"` (default) loads
        the most recently registered version. Auto-detects
        classifier/regressor from the saved file's own metadata (via
        `kanboost.ops.serving._load_any`), same as the serving layer."""
        manifest = self._load_manifest(name)
        if not manifest["versions"]:
            raise ValueError(f"no versions registered for {name!r}")
        if version == "latest":
            version = manifest["versions"][-1]["version"]
        if not any(v["version"] == version for v in manifest["versions"]):
            raise ValueError(f"no version {version} registered for {name!r}")

        from ..ops.serving import _load_any
        return _load_any(self._model_path(name, version))

    def list(self, name: str | None = None) -> list:
        """List registered versions for `name`, or every registered
        model name if `name` is omitted."""
        if name is not None:
            return self._load_manifest(name)["versions"]
        return [
            fname[: -len("_manifest.json")]
            for fname in os.listdir(self.root_dir)
            if fname.endswith("_manifest.json")
        ]

    def push(self, name: str, version="latest", *, bucket: str, api_key: str | None = None,
              base_url: str | None = None) -> dict:
        """Push a registered version to the remote object store --
        delegates to the existing `kanboost.registry.mlhub.push_model`,
        not a reimplementation."""
        if version == "latest":
            manifest = self._load_manifest(name)
            if not manifest["versions"]:
                raise ValueError(f"no versions registered for {name!r}")
            version = manifest["versions"][-1]["version"]

        from .mlhub import push_model
        path = self._model_path(name, version)
        return push_model(path, bucket, key=f"{name}_v{version}.pt", api_key=api_key, base_url=base_url)
