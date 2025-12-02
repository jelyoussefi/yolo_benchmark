#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
YOLO Model Benchmark Script
Benchmarks YOLO models using Intel OpenVINO Runtime
"""

import os
import sys
import argparse
from pathlib import Path
from time import perf_counter
from typing import Tuple, List
import numpy as np
import cv2
from openvino.runtime import Core, CompiledModel

# ANSI Color Codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# COCO class labels
LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def print_header():
    """Print a stylized header"""
    print()
    print(f"{Colors.CYAN}{'─' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'YOLO Model Benchmark - Intel OpenVINO':^70}{Colors.END}")
    print(f"{Colors.CYAN}{'─' * 70}{Colors.END}")


def print_config(model_name: str, precision: str, device: str, input_path: str):
    """Print configuration details"""
    print(f"{Colors.CYAN}{'─' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'Configuration':^70}{Colors.END}")
    print(f"{Colors.CYAN}{'─' * 70}{Colors.END}")
    print(f"\t{Colors.YELLOW}Model:{Colors.END}     {model_name}")
    print(f"\t{Colors.YELLOW}Precision:{Colors.END} {precision}")
    print(f"\t{Colors.YELLOW}Device:{Colors.END}    {device}")
    print(f"\t{Colors.YELLOW}Input:{Colors.END}     {input_path}")


def print_results(fps: float, avg_latency_ms: float, output_path: str):
    """Print benchmark results with colors"""
    print(f"{Colors.CYAN}{'─' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'Result':^70}{Colors.END}")
    print(f"{Colors.CYAN}{'─' * 70}{Colors.END}")
    
    fps_str = f"{fps:.2f}"
    print(f"\t{Colors.GREEN}FPS:{Colors.END}      {fps_str}")
    print(f"\t{Colors.GREEN}Latency:{Colors.END}  {avg_latency_ms:>{len(fps_str)}.2f} ms")
    print(f"\t{Colors.GREEN}Output:{Colors.END}   {output_path}")
    print(f"{Colors.CYAN}{'─' * 70}{Colors.END}")
    print()


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}✗ Error: {message}{Colors.END}")


def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ Warning: {message}{Colors.END}")


def print_info(message: str):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {message}{Colors.END}")


class YoloBenchmark:
    """YOLO Benchmark class using OpenVINO"""
    
    def __init__(self, model_path: str, device: str = "CPU", 
                 conf_threshold: float = 0.3, iou_threshold: float = 0.5):
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.labels = LABELS
        self.num_masks = 32
        
        # Initialize OpenVINO
        self.core = Core()
        self._load_model()
    
    def _load_model(self):
        """Load and compile the OpenVINO model"""
        # Read model
        self.model = self.core.read_model(self.model_path)
        
        # Get input shape
        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        
        # Handle dynamic shapes - check if dimensions are Dimension objects or ints
        h_dim = self.input_shape[2]
        w_dim = self.input_shape[3]
        
        # Check if dynamic (Dimension object with is_dynamic) or static (int)
        h_is_dynamic = hasattr(h_dim, 'is_dynamic') and h_dim.is_dynamic
        w_is_dynamic = hasattr(w_dim, 'is_dynamic') and w_dim.is_dynamic
        
        if h_is_dynamic or w_is_dynamic:
            self.input_height = 640
            self.input_width = 640
        else:
            self.input_height = int(h_dim)
            self.input_width = int(w_dim)
        
        # Compile model for specific device
        self.compiled_model = self.core.compile_model(self.model, self.device)
        self.infer_request = self.compiled_model.create_infer_request()
        
        # Get output layer(s) - check for segmentation model
        self.output_layer = self.compiled_model.output(0)
        
        # Check if this is a segmentation model (has 2 outputs: detections + mask protos)
        self.is_segmentation = len(self.compiled_model.outputs) >= 2
        if self.is_segmentation:
            self.mask_protos_layer = self.compiled_model.output(1)
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Preprocess image for inference"""
        original_shape = image.shape[:2]
        
        # Resize with letterboxing and get transformation info
        resized, transform_info = self._letterbox(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        resized = resized.astype(np.float32) / 255.0
        
        # HWC to CHW format
        resized = resized.transpose(2, 0, 1)
        
        # Add batch dimension
        resized = np.expand_dims(resized, axis=0)
        
        return resized, transform_info
    
    def _letterbox(self, image: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, dict]:
        """Resize image with letterboxing to maintain aspect ratio"""
        h, w = image.shape[:2]
        new_w, new_h = new_shape
        
        # Calculate scaling factor
        scale = min(new_w / w, new_h / h)
        
        # New dimensions
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas and paste
        canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
        
        # Center the image
        pad_top = (new_h - scaled_h) // 2
        pad_left = (new_w - scaled_w) // 2
        canvas[pad_top:pad_top + scaled_h, pad_left:pad_left + scaled_w] = resized
        
        # Store transformation info for reverse mapping
        transform_info = {
            'original_shape': (h, w),
            'scale': scale,
            'pad_top': pad_top,
            'pad_left': pad_left
        }
        
        return canvas, transform_info
    
    def infer(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference, returns (detections, mask_protos) for seg models"""
        self.infer_request.infer({self.input_layer.any_name: input_tensor})
        output = self.infer_request.get_output_tensor(0).data.copy()
        
        mask_protos = None
        if self.is_segmentation:
            mask_protos = self.infer_request.get_output_tensor(1).data.copy()
        
        return output, mask_protos
    
    def postprocess(self, output: np.ndarray, original_image: np.ndarray, 
                    transform_info: dict, mask_protos: np.ndarray = None) -> Tuple[np.ndarray, int]:
        """Postprocess model output and draw detections/segmentations"""
        predictions = np.squeeze(output).T
        
        # Determine number of classes (accounting for mask coefficients if present)
        if self.is_segmentation:
            num_classes = output.shape[1] - self.num_masks - 4
        else:
            num_classes = output.shape[1] - 4
        
        if num_classes <= 0:
            num_classes = output.shape[1] - 4
        
        # Filter by confidence
        scores = np.max(predictions[:, 4:4 + num_classes], axis=1)
        mask = scores > self.conf_threshold
        predictions = predictions[mask]
        scores = scores[mask]
        
        if len(scores) == 0:
            return original_image, 0
        
        # Get class IDs
        class_ids = np.argmax(predictions[:, 4:4 + num_classes], axis=1)
        
        # Get boxes
        boxes = self._get_boxes(predictions, original_image, transform_info)
        
        # Apply NMS
        indices = self._nms(boxes, scores, self.iou_threshold)
        
        boxes = boxes[indices]
        scores = scores[indices]
        class_ids = class_ids[indices]
        
        # Handle segmentation masks if available
        masks = None
        if self.is_segmentation and mask_protos is not None:
            # Get mask coefficients for filtered predictions
            mask_coeffs = predictions[indices, 4 + num_classes:4 + num_classes + self.num_masks]
            masks = self._process_masks(mask_protos, mask_coeffs, boxes, original_image.shape, transform_info)
        
        # Draw detections (and masks if segmentation)
        result_image = self._draw_detections(original_image, boxes, scores, class_ids, masks)
        
        return result_image, len(indices)
    
    def _get_boxes(self, predictions: np.ndarray, orig_img: np.ndarray,
                   transform_info: dict) -> np.ndarray:
        """Extract and scale bounding boxes, reversing letterbox transformation"""
        img_h, img_w = orig_img.shape[:2]
        boxes = predictions[:, :4].copy()
        
        # Get transformation parameters
        scale = transform_info['scale']
        pad_top = transform_info['pad_top']
        pad_left = transform_info['pad_left']
        
        # Boxes are in xywh format relative to input size (e.g., 640x640)
        # First convert xywh to xyxy
        x_center = boxes[:, 0]
        y_center = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Remove padding offset (boxes are in padded coordinate space)
        x1 = x1 - pad_left
        y1 = y1 - pad_top
        x2 = x2 - pad_left
        y2 = y2 - pad_top
        
        # Scale back to original image size
        x1 = x1 / scale
        y1 = y1 / scale
        x2 = x2 / scale
        y2 = y2 / scale
        
        # Clip to image bounds
        x1 = np.clip(x1, 0, img_w)
        y1 = np.clip(y1, 0, img_h)
        x2 = np.clip(x2, 0, img_w)
        y2 = np.clip(y2, 0, img_h)
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        return boxes
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, 
             iou_threshold: float) -> List[int]:
        """Non-maximum suppression"""
        sorted_indices = np.argsort(scores)[::-1]
        keep = []
        
        while sorted_indices.size > 0:
            idx = sorted_indices[0]
            keep.append(idx)
            
            if sorted_indices.size == 1:
                break
            
            ious = self._compute_iou(boxes[idx], boxes[sorted_indices[1:]])
            mask = ious < iou_threshold
            sorted_indices = sorted_indices[1:][mask]
        
        return keep
    
    def _compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU between a box and array of boxes"""
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        
        return intersection / union
    
    def _process_masks(self, mask_protos: np.ndarray, mask_coeffs: np.ndarray,
                       boxes: np.ndarray, orig_shape: Tuple[int, int, int],
                       transform_info: dict) -> List[np.ndarray]:
        """Process segmentation masks from mask prototypes and coefficients"""
        # mask_protos shape: (1, 32, H, W) or (32, H, W)
        # mask_coeffs shape: (N, 32)
        
        protos = np.squeeze(mask_protos)  # (32, H, W)
        if protos.ndim == 2:
            protos = protos[np.newaxis, ...]
            
        proto_h, proto_w = protos.shape[1], protos.shape[2]
        orig_h, orig_w = orig_shape[:2]
        
        masks = []
        for i, (coeffs, box) in enumerate(zip(mask_coeffs, boxes)):
            # Matrix multiplication: coeffs @ protos -> (H, W)
            mask = np.tensordot(coeffs, protos, axes=([0], [0]))  # (proto_h, proto_w)
            
            # Apply sigmoid
            mask = 1 / (1 + np.exp(-mask))
            
            # Resize mask to input size (e.g., 640x640)
            mask = cv2.resize(mask, (self.input_width, self.input_height))
            
            # Reverse letterbox transformation
            scale = transform_info['scale']
            pad_top = transform_info['pad_top']
            pad_left = transform_info['pad_left']
            
            # Remove padding
            scaled_h = int(orig_h * scale)
            scaled_w = int(orig_w * scale)
            mask = mask[pad_top:pad_top + scaled_h, pad_left:pad_left + scaled_w]
            
            # Resize to original image size
            mask = cv2.resize(mask, (orig_w, orig_h))
            
            # Crop mask to bounding box for efficiency
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_w, x2)
            y2 = min(orig_h, y2)
            
            # Create full mask zeroed outside bbox
            full_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
            full_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            
            # Threshold
            full_mask = (full_mask > 0.5).astype(np.uint8)
            
            masks.append(full_mask)
        
        return masks
    
    def _draw_detections(self, image: np.ndarray, boxes: np.ndarray, 
                         scores: np.ndarray, class_ids: np.ndarray,
                         masks: List[np.ndarray] = None) -> np.ndarray:
        """Draw bounding boxes, labels, and segmentation masks on image"""
        result = image.copy()
        
        # Draw masks first (semi-transparent overlay)
        if masks is not None:
            for i, mask in enumerate(masks):
                color = self._get_color(int(class_ids[i]))
                # Direct alpha blending for true color representation
                alpha = 0.5
                mask_bool = mask > 0
                result[mask_bool] = (
                    (1 - alpha) * result[mask_bool] + alpha * np.array(color)
                ).astype(np.uint8)
        
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Generate color based on class
            color = self._get_color(int(class_id))
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{self.labels[int(class_id)]}: {score:.2f}"
            
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            
            # Draw label background
            cv2.rectangle(result, (x1, y1 - text_h - 10), 
                         (x1 + text_w + 4, y1), color, -1)
            
            # Draw text
            cv2.putText(result, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return result
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for a class ID using famous/standard colors for common classes"""
        # Predefined vibrant colors for common COCO classes (BGR format)
        CLASS_COLORS = {
            0: (255, 0, 0),       # person - Blue
            1: (0, 128, 255),     # bicycle - Orange
            2: (0, 255, 0),       # car - Green
            3: (0, 255, 255),     # motorcycle - Yellow
            4: (255, 255, 0),     # airplane - Cyan
            5: (255, 0, 255),     # bus - Magenta
            6: (0, 0, 255),       # train - Red
            7: (128, 0, 128),     # truck - Purple
            8: (255, 192, 203),   # boat - Pink
            9: (0, 255, 128),     # traffic light - Spring Green
            10: (255, 128, 0),    # fire hydrant - Sky Blue
            11: (0, 0, 128),      # stop sign - Maroon
            12: (128, 128, 0),    # parking meter - Teal
            13: (0, 128, 128),    # bench - Olive
            14: (180, 105, 255),  # bird - Hot Pink
            15: (255, 144, 30),   # cat - Dodger Blue
            16: (128, 0, 128),    # dog - Purple
            17: (255, 0, 255),    # horse - Magenta
            18: (147, 20, 255),   # sheep - Deep Pink
            19: (42, 42, 165),    # cow - Brown
        }
        
        if class_id in CLASS_COLORS:
            return CLASS_COLORS[class_id]
        
        # Generate pseudo-random colors for other classes
        np.random.seed(class_id + 100)
        return tuple(int(x) for x in np.random.randint(80, 255, 3))
    
    def benchmark(self, image: np.ndarray, warmup_runs: int = 10, 
                  benchmark_runs: int = 100) -> dict:
        """Run benchmark and return metrics"""
        # Preprocess
        input_tensor, transform_info = self.preprocess(image)
        
        # Warmup
        for _ in range(warmup_runs):
            _ = self.infer(input_tensor)
        
        # Benchmark
        latencies = []
        
        for i in range(benchmark_runs):
            start = perf_counter()
            output, mask_protos = self.infer(input_tensor)
            end = perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        # Get output with detections/segmentations
        result_image, num_detections = self.postprocess(output, image, transform_info, mask_protos)
        
        # Calculate metrics
        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        fps = 1000.0 / avg_latency
        
        return {
            'fps': fps,
            'avg_latency_ms': avg_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'first_latency_ms': latencies[0],
            'num_detections': num_detections,
            'result_image': result_image,
            'warmup_runs': warmup_runs,
            'benchmark_runs': benchmark_runs,
            'is_segmentation': self.is_segmentation
        }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Benchmark YOLO models with Intel OpenVINO',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Model name (e.g., yolo11n, yolov8s)'
    )
    
    parser.add_argument(
        '--precision', '-p',
        type=str,
        default='FP16',
        choices=['FP32', 'FP16', 'INT8'],
        help='Model precision'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input image path'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='CPU',
        help='Target device (CPU, GPU, NPU, AUTO)'
    )
    
    parser.add_argument(
        '--warmup', '-w',
        type=int,
        default=10,
        help='Number of warmup iterations'
    )
    
    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=100,
        help='Number of benchmark iterations'
    )
    
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.3,
        help='Confidence threshold for detections'
    )
    
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for NMS'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for result images'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print_header()
    
    # Build model path
    model_path = Path(f"/opt/models/{args.model}/{args.precision}/{args.model}.xml")
    
    # Check if model exists
    if not model_path.exists():
        print_error(f"Model not found: {model_path}")
        sys.exit(1)
    
    # Check if input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print_error(f"Input image not found: {input_path}")
        sys.exit(1)
    
    # Print configuration
    print_config(args.model, args.precision, args.device, str(input_path))
    
    # Load image
    image = cv2.imread(str(input_path))
    if image is None:
        print_error(f"Failed to load image: {input_path}")
        sys.exit(1)
    
    try:
        # Initialize benchmark
        benchmark = YoloBenchmark(
            model_path=str(model_path),
            device=args.device,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold
        )
        
        # Run benchmark
        results = benchmark.benchmark(
            image, 
            warmup_runs=args.warmup, 
            benchmark_runs=args.iterations
        )
        
        # Save output image
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.model}.jpg"
        cv2.imwrite(str(output_path), results['result_image'])
        
        # Print results
        print_results(
            fps=results['fps'],
            avg_latency_ms=results['avg_latency_ms'],
            output_path=str(output_path)
        )
        
    except Exception as e:
        print_error(str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
