import tensorflow as tf
import numpy as np
import os
import json
import joblib
from datetime import datetime
import matplotlib.pyplot as plt


class ModelOptimizer:
    """
    TensorFlow Lite model optimizer for mobile deployment
    Converts trained models to lightweight, mobile-ready formats
    """

    def __init__(self, model_path, class_names_path):
        """
        Initialize the model optimizer

        Args:
            model_path: Path to the trained model
            class_names_path: Path to class names JSON
        """
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_names = None
        self.load_model_and_classes()

    def load_model_and_classes(self):
        """Load the trained model and class names"""
        print("🔄 Loading model for optimization...")
        try:
            self.model = joblib.load(self.model_path)
            with open(self.class_names_path, "r") as f:
                self.class_names = json.load(f)
            print(f"✅ Model loaded: {os.path.basename(self.model_path)}")
            print(f"📚 Classes: {len(self.class_names)} categories")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def get_model_size(self, model_path):
        """Get model file size in MB"""
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb

    def convert_to_tflite(self, output_path, optimization_type="default"):
        """
        Convert Keras model to TensorFlow Lite format

        Args:
            output_path: Path to save the TFLite model
            optimization_type: Type of optimization ('default', 'size', 'speed')

        Returns:
            dict: Conversion results and statistics
        """
        print(
            f"\n🔄 Converting to TensorFlow Lite ({optimization_type} optimization)..."
        )

        # Create TensorFlow Lite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Apply optimization based on type
        if optimization_type == "default":
            # Basic optimization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        elif optimization_type == "size":
            # Maximum size reduction
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        elif optimization_type == "speed":
            # Optimized for inference speed
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]

        # Convert the model
        try:
            tflite_model = converter.convert()

            # Save the model
            os.makedirs(
                os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True,
            )
            with open(output_path, "wb") as f:
                f.write(tflite_model)

            # Get statistics
            original_size = self.get_model_size(self.model_path)
            tflite_size = self.get_model_size(output_path)
            compression_ratio = original_size / tflite_size

            results = {
                "success": True,
                "optimization_type": optimization_type,
                "original_size_mb": original_size,
                "tflite_size_mb": tflite_size,
                "compression_ratio": compression_ratio,
                "size_reduction_percent": (
                    (original_size - tflite_size) / original_size
                )
                * 100,
                "output_path": output_path,
            }

            print(f"✅ TFLite conversion completed!")
            print(f"📦 Original size: {original_size:.2f} MB")
            print(f"📦 TFLite size: {tflite_size:.2f} MB")
            print(f"🎯 Compression ratio: {compression_ratio:.2f}x")
            print(f"💾 Size reduction: {results['size_reduction_percent']:.1f}%")

            return results

        except Exception as e:
            print(f"❌ TFLite conversion failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "optimization_type": optimization_type,
            }

    def benchmark_tflite_model(self, tflite_path, num_runs=100):
        """
        Benchmark TensorFlow Lite model performance

        Args:
            tflite_path: Path to TFLite model
            num_runs: Number of inference runs for benchmarking

        Returns:
            dict: Performance metrics
        """
        print(f"\n⚡ Benchmarking TFLite model: {os.path.basename(tflite_path)}")

        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"📊 Input shape: {input_details[0]['shape']}")
        print(f"📊 Output shape: {output_details[0]['shape']}")

        # Prepare synthetic test data
        input_shape = input_details[0]["shape"]
        test_data = [
            np.random.random(input_shape).astype(np.float32) for _ in range(10)
        ]

        # Warm up
        print("🔥 Warming up model...")
        for i in range(5):
            interpreter.set_tensor(input_details[0]["index"], test_data[0])
            interpreter.invoke()

        # Benchmark inference time
        print(f"⏱️ Running {num_runs} inference cycles...")
        import time

        times = []
        for i in range(num_runs):
            # Use different test images cyclically
            test_input = test_data[i % len(test_data)]

            start_time = time.time()
            interpreter.set_tensor(input_details[0]["index"], test_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]["index"])
            end_time = time.time()

            times.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)

        results = {
            "model_path": tflite_path,
            "num_runs": num_runs,
            "avg_inference_ms": avg_time,
            "min_inference_ms": min_time,
            "max_inference_ms": max_time,
            "std_inference_ms": std_time,
            "fps": 1000 / avg_time,  # Frames per second
            "input_shape": input_details[0]["shape"].tolist(),
            "output_shape": output_details[0]["shape"].tolist(),
            "model_size_mb": self.get_model_size(tflite_path),
        }

        print(f"✅ Benchmark completed!")
        print(f"⚡ Average inference time: {avg_time:.2f} ms")
        print(f"📱 Estimated FPS: {results['fps']:.1f}")
        print(f"📦 Model size: {results['model_size_mb']:.2f} MB")

        return results

    def create_optimization_comparison(self, output_dir="optimization"):
        """
        Create all optimized versions and compare them

        Args:
            output_dir: Directory to save optimized models

        Returns:
            dict: Comparison results
        """
        print("\n🚀 Creating comprehensive model optimization comparison...")
        os.makedirs(output_dir, exist_ok=True)

        optimization_types = ["default", "size", "speed"]
        results = {}

        # Convert to different TFLite formats
        for opt_type in optimization_types:
            output_path = os.path.join(
                output_dir, f"animal_classifier_{opt_type}.tflite"
            )

            print(f"\n{'='*15} {opt_type.upper()} OPTIMIZATION {'='*15}")
            conversion_result = self.convert_to_tflite(output_path, opt_type)

            if conversion_result["success"]:
                # Benchmark the converted model
                benchmark_result = self.benchmark_tflite_model(output_path)

                # Combine results
                results[opt_type] = {
                    "conversion": conversion_result,
                    "benchmark": benchmark_result,
                }
            else:
                results[opt_type] = {"conversion": conversion_result, "benchmark": None}

        # Create comparison visualization
        self._create_comparison_plots(results, output_dir)

        # Save detailed results
        self._save_optimization_report(results, output_dir)

        return results

    def _create_comparison_plots(self, results, output_dir):
        """Create visualization comparing different optimizations"""

        # Extract successful results
        successful_results = {
            k: v
            for k, v in results.items()
            if v["conversion"]["success"] and v["benchmark"]
        }

        if len(successful_results) < 2:
            print("⚠️ Not enough successful conversions for comparison plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "📊 TensorFlow Lite Model Optimization Comparison",
            fontsize=16,
            fontweight="bold",
        )

        # Extract data for plotting
        opt_types = list(successful_results.keys())
        sizes = [
            successful_results[opt]["conversion"]["tflite_size_mb"] for opt in opt_types
        ]
        inference_times = [
            successful_results[opt]["benchmark"]["avg_inference_ms"]
            for opt in opt_types
        ]
        fps_values = [successful_results[opt]["benchmark"]["fps"] for opt in opt_types]
        compression_ratios = [
            successful_results[opt]["conversion"]["compression_ratio"]
            for opt in opt_types
        ]

        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

        # Model Size Comparison
        bars1 = axes[0, 0].bar(
            opt_types, sizes, color=colors[: len(opt_types)], alpha=0.8
        )
        axes[0, 0].set_title("📦 Model Size Comparison", fontweight="bold")
        axes[0, 0].set_ylabel("Size (MB)")
        for bar, size in zip(bars1, sizes):
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{size:.1f}MB",
                ha="center",
                fontweight="bold",
            )

        # Inference Time Comparison
        bars2 = axes[0, 1].bar(
            opt_types, inference_times, color=colors[: len(opt_types)], alpha=0.8
        )
        axes[0, 1].set_title("⚡ Inference Time Comparison", fontweight="bold")
        axes[0, 1].set_ylabel("Time (ms)")
        for bar, time in zip(bars2, inference_times):
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{time:.1f}ms",
                ha="center",
                fontweight="bold",
            )

        # FPS Comparison
        bars3 = axes[1, 0].bar(
            opt_types, fps_values, color=colors[: len(opt_types)], alpha=0.8
        )
        axes[1, 0].set_title("📱 Frames Per Second (FPS)", fontweight="bold")
        axes[1, 0].set_ylabel("FPS")
        for bar, fps in zip(bars3, fps_values):
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{fps:.1f}",
                ha="center",
                fontweight="bold",
            )

        # Compression Ratio
        bars4 = axes[1, 1].bar(
            opt_types, compression_ratios, color=colors[: len(opt_types)], alpha=0.8
        )
        axes[1, 1].set_title("🗜️ Compression Ratio", fontweight="bold")
        axes[1, 1].set_ylabel("Ratio (Original/Optimized)")
        for bar, ratio in zip(bars4, compression_ratios):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{ratio:.1f}x",
                ha="center",
                fontweight="bold",
            )

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"optimization_comparison_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.savefig(
            os.path.join(output_dir, "optimization_comparison_latest.png"),
            dpi=300,
            bbox_inches="tight",
        )

        print(f"📊 Comparison plots saved to: {plot_path}")
        plt.show()

    def _save_optimization_report(self, results, output_dir):
        """Save detailed optimization report"""

        report_path = os.path.join(output_dir, "optimization_report.json")

        # Add metadata
        report = {
            "timestamp": datetime.now().isoformat(),
            "original_model": self.model_path,
            "original_size_mb": self.get_model_size(self.model_path),
            "class_names": self.class_names,
            "optimization_results": results,
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"📄 Detailed report saved to: {report_path}")


def optimize_animal_classification_models():
    """Main function to optimize all animal classification models"""

    print("🚀 Animal Classification Model Optimization Pipeline")
    print("=" * 60)

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Model paths (relative to script directory)
    models_to_optimize = [
        ("aniClass_EFF_Stage2.pkl", "EfficientNet Stage 2 (Best Model)"),
        ("aniClass_EFF_Stage1.pkl", "EfficientNet Stage 1"),
        ("aniClass_CNN_enhanced.pkl", "Custom CNN"),
    ]

    class_names_path = os.path.join(script_dir, "class_names.json")

    # Check if class names exist
    if not os.path.exists(class_names_path):
        print(f"❌ Class names file not found: {class_names_path}")
        print("💡 Train your model first to generate class_names.json")
        return

    results_summary = {}

    for model_file, model_name in models_to_optimize:
        model_path = os.path.join(script_dir, model_file)

        if os.path.exists(model_path):
            print(f"\n{'='*20} {model_name} {'='*20}")

            try:
                # Create optimizer
                optimizer = ModelOptimizer(model_path, class_names_path)

                # Create optimization comparison
                output_dir = os.path.join(
                    script_dir, f"optimization_{os.path.splitext(model_file)[0]}"
                )
                results = optimizer.create_optimization_comparison(output_dir)

                results_summary[model_name] = {
                    "model_file": model_file,
                    "output_dir": output_dir,
                    "results": results,
                }

            except Exception as e:
                print(f"❌ Failed to optimize {model_name}: {str(e)}")

        else:
            print(f"⚠️ Model file not found: {model_path}")

    # Create summary report
    if results_summary:
        print(f"\n{'='*60}")
        print("📊 OPTIMIZATION SUMMARY")
        print(f"{'='*60}")

        for model_name, data in results_summary.items():
            print(f"\n🎯 {model_name}")
            if "results" in data:
                for opt_type, result in data["results"].items():
                    if result["conversion"]["success"]:
                        size_mb = result["conversion"]["tflite_size_mb"]
                        compression = result["conversion"]["compression_ratio"]
                        if result["benchmark"]:
                            fps = result["benchmark"]["fps"]
                            avg_ms = result["benchmark"]["avg_inference_ms"]
                            print(
                                f"   {opt_type.capitalize()}: {size_mb:.1f}MB ({compression:.1f}x smaller), {avg_ms:.1f}ms ({fps:.1f} FPS)"
                            )
                        else:
                            print(
                                f"   {opt_type.capitalize()}: {size_mb:.1f}MB ({compression:.1f}x smaller)"
                            )

        print(f"\n✅ Optimization completed!")
        print(f"📊 Check individual optimization folders for detailed results")
        print(f"💡 Use the TFLite models in your app for better performance!")


if __name__ == "__main__":
    optimize_animal_classification_models()
