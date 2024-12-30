import streamlit as st
import torch
import os
import sys

def check_gpu_details():
    st.write("System Information:")
    st.code(f"Python version: {sys.version}")
    st.code(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        st.success("✅ CUDA is available")
        st.info(f"GPU: {torch.cuda.get_device_name(0)}")
        st.info(f"CUDA version: {torch.version.cuda}")
        
        # Show memory information
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        st.info(f"Total GPU memory: {total_memory:.2f} GB")
        st.info(f"Currently allocated: {allocated_memory:.2f} GB")
    else:
        st.error("❌ CUDA is not available")
        
        # Check environment variables
        cuda_path = os.environ.get('CUDA_PATH')
        st.write("CUDA_PATH:", cuda_path)
        
        # Check PATH for CUDA
        path = os.environ.get('PATH')
        cuda_in_path = any('cuda' in p.lower() for p in path.split(os.pathsep))
        st.write("CUDA in PATH:", cuda_in_path)

# Add this to your Streamlit app
if __name__ == "__main__":
    st.title("GPU Configuration Check")
    check_gpu_details()
