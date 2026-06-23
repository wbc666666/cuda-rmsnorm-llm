from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rmsnorm_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="rmsnorm_cuda_ext",
            sources=[
                "src/binding.cpp",
                "src/rmsnorm_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)