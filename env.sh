export GEM5_PATH=$(realpath $(pwd)/..)
export PYTHONPATH=$GEM5_PATH/configs:$(pwd)/common/configs
export M5_PATH=$(pwd)/common/system/arm:$(pwd)/common/system/riscv
export ARCHI_PATH=$(dirname "$0")