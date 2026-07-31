"""
kanboost is a pure-Python package by default (see pyproject.toml for
metadata/dependencies) -- this file exists ONLY to attempt building one
OPTIONAL native extension, kanboost._ga2m_cpp, which accelerates
kanboost.train.ga2m's predict-time forward pass (~2.26x faster than the
numba-accelerated Python path, measured on INSPIRE; see
docs/inspire_kanboost_evaluation.md and AI_REVIEW_LOOP.md, CC-21).

This extension is entirely optional and best-effort:
  - Requires `pybind11` to be installed at BUILD time (not a runtime
    dependency -- `pip install pybind11` before building from source, or
    `pip install kanboost[cppext]` documents the same requirement).
  - Requires a C++17 compiler.
  - If either is missing, or the compile step fails for any reason, the
    build falls back to a pure-Python install automatically -- kanboost
    works identically either way, just without this specific speedup.
    kanboost.train.ga2m auto-detects the compiled extension at import
    time and uses it only if present.

PyPI wheels published via `.github/workflows/publish.yml` remain
pure-Python (this extension is not built as part of that pipeline) --
build from source with `pybind11` pre-installed to get the accelerated
path.
"""
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

ext_modules = []
cmdclass = {}

try:
    import pybind11

    from setuptools import Extension

    extra_compile_args = ["-std=c++17", "-O3"]
    extra_link_args = []
    # -ffast-math/-march=native are GCC/Clang-only; MSVC uses different
    # flags entirely, so leave MSVC on its defaults (still gets -O2 via
    # setuptools' own default optimization flag) rather than guessing.
    import sys
    if sys.platform != "win32" or True:
        # These are appended unconditionally; the custom build_ext below
        # strips them out again for the MSVC compiler specifically.
        extra_compile_args += ["-ffast-math"]

    ext_modules = [
        Extension(
            "kanboost._ga2m_cpp",
            sources=["kanboost/_native/ga2m_ext.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            optional=True,
        )
    ]

    class OptionalBuildExt(_build_ext):
        """Never let a failure to build this optional extension fail the
        whole install -- fall back to pure-Python kanboost silently
        (with a warning), exactly as if pybind11/a compiler were absent."""

        def build_extension(self, ext):
            # MSVC doesn't understand GCC/Clang flags like -ffast-math;
            # drop them when the active compiler is MSVC.
            if self.compiler.compiler_type == "msvc":
                ext.extra_compile_args = [
                    a for a in ext.extra_compile_args if not a.startswith("-f") and not a.startswith("-march")
                ]
                ext.extra_compile_args += ["/std:c++17", "/O2"]
            try:
                super().build_extension(ext)
            except Exception as exc:  # noqa: BLE001 -- intentionally broad, see module docstring
                print(
                    f"WARNING: could not build optional kanboost._ga2m_cpp extension ({exc!r}); "
                    "falling back to the pure-Python/numba predict path. This is not an error -- "
                    "kanboost works fully without it."
                )

    cmdclass = {"build_ext": OptionalBuildExt}

except ImportError:
    pass  # pybind11 not installed at build time -- ship pure-Python only, silently.

setup(ext_modules=ext_modules, cmdclass=cmdclass)
