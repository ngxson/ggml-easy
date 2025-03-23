import gguf
import argparse
import logging
import sys
import torch
import json
import os
import numpy as np
from typing import cast, ContextManager, Any, Iterator
from pathlib import Path
from torch import Tensor

# some tensor names are too long, ggml refuses to load them
# this function renames them to shorter names
def rename_tensor(name: str) -> str:
    replacements = {
        "quantizer.acoustic_residual_vector_quantizer": "quantizer.acoustic_rvq", # kyutai mimi
        "quantizer.semantic_residual_vector_quantizer": "quantizer.semantic_rvq", # kyutai mimi
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name

# (copied from convert_hf_to_gguf.py)
# tree of lazy tensors
class LazyTorchTensor(gguf.LazyBase):
    _tensor_type = torch.Tensor
    # to keep the type-checker happy
    dtype: torch.dtype
    shape: torch.Size

    # only used when converting a torch.Tensor to a np.ndarray
    _dtype_map: dict[torch.dtype, type] = {
        torch.float16: np.float16,
        torch.float32: np.float32,
    }

    # used for safetensors slices
    # ref: https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/src/lib.rs#L1046
    # TODO: uncomment U64, U32, and U16, ref: https://github.com/pytorch/pytorch/issues/58734
    _dtype_str_map: dict[str, torch.dtype] = {
        "F64": torch.float64,
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        # "U64": torch.uint64,
        "I64": torch.int64,
        # "U32": torch.uint32,
        "I32": torch.int32,
        # "U16": torch.uint16,
        "I16": torch.int16,
        "U8": torch.uint8,
        "I8": torch.int8,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
    }

    def numpy(self) -> gguf.LazyNumpyTensor:
        dtype = self._dtype_map[self.dtype]
        return gguf.LazyNumpyTensor(
            meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(dtype, self.shape),
            args=(self,),
            func=(lambda s: s.numpy())
        )

    @classmethod
    def meta_with_dtype_and_shape(cls, dtype: torch.dtype, shape: tuple[int, ...]) -> Tensor:
        return torch.empty(size=shape, dtype=dtype, device="meta")

    @classmethod
    def from_safetensors_slice(cls, st_slice: Any) -> Tensor:
        dtype = cls._dtype_str_map[st_slice.get_dtype()]
        shape: tuple[int, ...] = tuple(st_slice.get_shape())
        lazy = cls(meta=cls.meta_with_dtype_and_shape(dtype, shape), args=(st_slice,), func=lambda s: s[:])
        return cast(torch.Tensor, lazy)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.numpy:
            return args[0].numpy()

        return cls._wrap_fn(func)(*args, **kwargs)

class Converter:
    in_file: Path
    out_file: Path
    ftype: gguf.LlamaFileType
    gguf_writer: gguf.GGUFWriter

    def __init__(self, in_file: Path, out_file: Path, ftype: gguf.LlamaFileType):
        self.in_file = in_file
        self.out_file = out_file
        self.ftype = ftype
        endianess = gguf.GGUFEndian.LITTLE
        self.gguf_writer = gguf.GGUFWriter(path=None, arch="unknown", endianess=endianess)

    def convert(self):
        print(f"Converting {self.in_file} to {self.out_file} with {self.ftype} data type.")

        for name, data_torch in self.get_tensors():
            old_dtype = data_torch.dtype
            is_1d = len(data_torch.shape) == 1
            can_quantize = not is_1d

            data_qtype = gguf.GGMLQuantizationType.F32
            if can_quantize:
                if self.ftype == gguf.LlamaFileType.ALL_F32:
                    data_qtype = gguf.GGMLQuantizationType.F32
                elif self.ftype == gguf.LlamaFileType.MOSTLY_F16:
                    data_qtype = gguf.GGMLQuantizationType.F16
                elif self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                    data_qtype = gguf.GGMLQuantizationType.BF16
                elif self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                    data_qtype = gguf.GGMLQuantizationType.Q8_0
                else:
                    raise ValueError(f"Unsupported file type: {self.ftype}")
                
            data = data_torch.numpy()
            try:
                data = gguf.quants.quantize(data, data_qtype)
            except Exception as e:
                print(f"Error quantizing tensor '{name}': {e}, fallback to F16")
                data_qtype = gguf.GGMLQuantizationType.F16
                data = gguf.quants.quantize(data, data_qtype)

            name = rename_tensor(name)

            # reverse shape to make it similar to the internal ggml dimension order
            shape_str = f"{{{', '.join(str(n) for n in reversed(data_torch.shape))}}}"
            print(f"{f'%-32s' % f'{name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

            self.gguf_writer.add_tensor(name, data, raw_dtype=data_qtype)

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        # TODO: support multiple shards in the future
        from safetensors import safe_open
        ctx = cast(ContextManager[Any], safe_open(self.in_file, framework="pt", device="cpu"))
        with ctx as model_part:
            for name in model_part.keys():
                data = model_part.get_slice(name)
                data = LazyTorchTensor.from_safetensors_slice(data)
                yield name, data

    def write(self):
        self.gguf_writer.write_header_to_file(path=self.out_file)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Convert safetensors to GGUF format.")
    parser.add_argument(
        "--outtype",
        choices=["f32", "f16", "bf16", "q8_0"],
        default="f32",
        help="Output data type (default: f32)"
    )
    parser.add_argument(
        "input_file", type=Path,
        help="Path to the input file (required)"
    )
    parser.add_argument(
        "output_file", type=Path,
        nargs="?",
        help="Path to the output file (optional). Default to input file with .gguf extension"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    }

    if args.outtype not in ftype_map:
        raise ValueError(f"Unsupported output data type: {args.outtype}")
    
    if args.output_file is None:
        args.output_file = args.input_file.with_suffix(".gguf")

    converter = Converter(args.input_file, args.output_file, ftype_map[args.outtype])
    converter.convert()
    converter.write()
