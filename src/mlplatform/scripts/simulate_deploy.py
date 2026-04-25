from __future__ import annotations

from mlplatform.registry import ModelRegistry


def main() -> None:
    registry = ModelRegistry()
    versions = registry.get_model_versions("sanity-model")
    production = registry.get_production_version("sanity-model")
    print({"deployed_versions": [version.version for version in versions], "production": production.version if production else None})


if __name__ == "__main__":
    main()
