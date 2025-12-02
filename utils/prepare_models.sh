#!/bin/bash
# =============================================================================
# YOLO Model Exporter for OpenVINO (FP32 & FP16)
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $(basename "$0") <model1> [model2] [model3] ...

Export YOLO models to OpenVINO format (FP32, FP16, and INT8).

Arguments:
    model    Model name (e.g., yolo11n) or path to .pt file

Examples:
    $(basename "$0") yolo11n yolo11s yolo11m
    $(basename "$0") ./custom_model.pt yolo11n
    $(basename "$0") yolo11n-seg

EOF
    exit 1
}

# -----------------------------------------------------------------------------
# Export single model
# -----------------------------------------------------------------------------
export_model() {
    local model="$1"
    local base_name="${model%.pt}"
    local dir_name
    dir_name=$(basename "${base_name}")

    echo "========================================"
    echo "Processing: ${model}"
    echo "========================================"

    mkdir -p "./${dir_name}/FP32" "./${dir_name}/FP16" "./${dir_name}/INT8"

    # FP32 export
    echo "[FP32] Exporting..."
    yolo export model="${model}" format=openvino half=false imgsz=640,640
    mv "${base_name}_openvino_model/${base_name}."* "./${dir_name}/FP32/"
    rm -rf "${base_name}_openvino_model"

    # FP16 export
    echo "[FP16] Exporting..."
    yolo export model="${model}" format=openvino half=true imgsz=640,640
    mv "${base_name}_openvino_model/${base_name}."* "./${dir_name}/FP16/"
    rm -rf "${base_name}_openvino_model"

    # INT8 export (uses different output folder naming: model_int8_openvino_model)
    echo "[INT8] Exporting..."
    yolo export model="${model}" format=openvino int8=true imgsz=640,640
    mv "${base_name}_int8_openvino_model/${base_name}."* "./${dir_name}/INT8/"
    rm -rf "${base_name}_int8_openvino_model"

    echo "[OK] ${dir_name} exported successfully"
    echo ""
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
[[ $# -eq 0 ]] && usage

echo "Starting model export..."
echo "Models: $*"
echo ""

for model in "$@"; do
    export_model "${model}"
done

echo "========================================"
echo "All models processed successfully!"
echo "========================================"
