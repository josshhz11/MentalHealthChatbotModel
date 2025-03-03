import torch
import platform
import os
import subprocess
import sys
import torch_directml
from torch import nn
import time

def get_detailed_gpu_info():
    try:
        # Get GPU information using Windows Management Instrumentation Command-line
        gpu_info = subprocess.check_output(
            "wmic path win32_VideoController get Name, AdapterRAM, DriverVersion",
            shell=True
        ).decode()
        return gpu_info.strip()
    except Exception as e:
        return f"Error getting GPU info: {str(e)}"

def check_gpu_support():
    print("=== System Information ===")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    
    print("\n=== GPU Information ===")
    print(get_detailed_gpu_info())
    
    print("\n=== PyTorch Device Configuration ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch default device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available - using CPU only")
    
    # Check for DirectML support
    try:
        print("\n=== DirectML Support (AMD GPU) ===")
        dml_device = torch_directml.device()
        if dml_device is not None:
            print("DirectML is available for AMD GPU acceleration")
            print(f"DirectML device: {dml_device}")
    except ImportError:
        print("\n=== DirectML Support (AMD GPU) ===")
        print("DirectML not installed. For AMD GPU support, run:")
        print("pip install torch-directml")

# Test GPU training with DirectML
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1000, 1000)
    
    def forward(self, x):
        return self.layer(x)

def test_gpu_training():
    print("=== GPU Training Test ===")
    
    # Create test data
    print("Creating test data...")
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)
    
    # Initialize model
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Test on CPU first
    print("\nTesting on CPU...")
    start_time = time.time()
    model.to('cpu')
    x_cpu, y_cpu = x.to('cpu'), y.to('cpu')
    
    for _ in range(100):
        optimizer.zero_grad()
        output = model(x_cpu)
        loss = criterion(output, y_cpu)
        loss.backward()
        optimizer.step()
    
    cpu_time = time.time() - start_time
    print(f"CPU Training Time: {cpu_time:.2f} seconds")
    
    # Test on DirectML device
    try:
        print("\nTesting on DirectML (AMD GPU)...")
        device = torch_directml.device()
        model.to(device)
        x_gpu, y_gpu = x.to(device), y.to(device)
        
        start_time = time.time()
        for _ in range(100):
            optimizer.zero_grad()
            output = model(x_gpu)
            loss = criterion(output, y_gpu)
            loss.backward()
            optimizer.step()
        
        gpu_time = time.time() - start_time
        print(f"DirectML Training Time: {gpu_time:.2f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"\nGPU Speedup: {speedup:.2f}x")
        
        # Memory usage
        print("\nMemory Usage:")
        print(f"GPU Tensor Memory: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")
        
        # Verify computation accuracy
        cpu_output = model.to('cpu')(x_cpu)
        gpu_output = model.to(device)(x_gpu).to('cpu')
        output_diff = torch.abs(cpu_output - gpu_output).mean().item()
        print(f"\nOutput Difference (CPU vs GPU): {output_diff:.6f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ GPU Test Failed: {str(e)}")
        return False
    finally:
        # Cleanup
        print("\nCleaning up...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.to('cpu')
        del x_gpu, y_gpu

if __name__ == "__main__":
    success = test_gpu_training()
    print(f"\nGPU Training Test {'Passed ✅' if success else 'Failed ❌'}")