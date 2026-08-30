import glob
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

src_dir = os.path.join(os.path.dirname(__file__), "src")

cu_sources = glob.glob(os.path.join(src_dir, "*.cu"))
cpp_sources = glob.glob(os.path.join(src_dir, "*.cpp"))
sources = sorted(cpp_sources + cu_sources)

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "-std=c++17",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
]

setup(
    name="warp_route",
    version="0.1.0",
    packages=["warp_route"],
    ext_modules=[
        CUDAExtension(
            name="warp_route._C",
            sources=sources,
            include_dirs=[src_dir],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch"],
)
