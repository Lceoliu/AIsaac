# HPC node (10.15.171.204:30575, 224 cores / 1.5 TB / H20): environment of the ws1 sandbox. source it.
export ABP_HOME=/101063/AIsaac/ws1
export HOME=$ABP_HOME/home TMPDIR=$ABP_HOME/tmp CUDA_CACHE_PATH=$ABP_HOME/cache
export ABP_BRIDGE_LUA=$ABP_HOME/bridge/abp_bridge.lua
export PATH=/opt/aisaac-env/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DISPLAY=:93
# one Xvfb for every instance (the engine needs a GLX-capable display at start-up; the stub list skips the rendering)
if ! pgrep -f "Xvfb :93" > /dev/null; then
  nohup Xvfb :93 -screen 0 640x480x24 -nolisten tcp -noreset > $ABP_HOME/runs/xvfb.log 2>&1 < /dev/null &
  sleep 1
fi
PY=/opt/aisaac-env/bin/python
cd $ABP_HOME/python
