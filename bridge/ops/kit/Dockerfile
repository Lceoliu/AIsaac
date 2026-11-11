# kit image: aisaac-bench:kit-<version>. Built OFFLINE by kit_build.sh from a context streamed by kit_ctx.py
# (Dockerfile, venv/, pydeps/; venv script paths already repaired in the stream). No RUN step, no network, no game
# files, no code, no credentials.
#   /opt/aisaac/venv    the 8x3090's shared venv (Python 3.10.12, torch 2.7.1+cu128), python -> /usr/bin/python3
#   /opt/aisaac/pydeps  bm1's overlay (numpy 2.2.6, zstandard 0.25.0, lz4, pytest, py-spy), first on sys.path
# PATH / PYTHONPATH give the package set bm1 runs use today (venv + PYTHONPATH=bm1/pydeps).
ARG KIT_BASE=aisaac-bench:20261019
FROM ${KIT_BASE}
ARG KIT_VERSION
ARG KIT_OPS_COMMIT
ARG KIT_BASE
COPY venv /opt/aisaac/venv
COPY pydeps /opt/aisaac/pydeps
ENV VIRTUAL_ENV=/opt/aisaac/venv \
    PYTHONPATH=/opt/aisaac/pydeps \
    PATH=/opt/aisaac/venv/bin:/opt/aisaac/pydeps/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    AISAAC_KIT_VERSION=${KIT_VERSION}
LABEL aisaac.kit.version="${KIT_VERSION}" \
      aisaac.kit.base_image="${KIT_BASE}" \
      aisaac.kit.ops_commit="${KIT_OPS_COMMIT}" \
      aisaac.kit.python="3.10.12" \
      aisaac.kit.torch="2.7.1+cu128" \
      aisaac.kit.cuda_runtime="12.8" \
      aisaac.kit.min_driver="525.60.13" \
      aisaac.kit.recommended_driver="570.26" \
      aisaac.kit.note="min_driver = CUDA 12 minor-version compatibility; >= 570.26 is the CUDA 12.8 driver"
WORKDIR /workspace
# the base image's CMD is an apt-get install of its own build; replace it with something harmless
ENTRYPOINT ["/bin/bash"]
CMD ["-c", "exec /opt/aisaac/venv/bin/python -V"]
