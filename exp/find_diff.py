import os
import numpy as np


musa_dump_dir = "debug_dumps_musa"
cpu_dump_dir = "debug_dumps_cpu"

musa_dump_tensor_paths = os.listdir(musa_dump_dir)
cpu_dump_tensor_paths = os.listdir(cpu_dump_dir)
musa_dump_tensor_paths.sort()
cpu_dump_tensor_paths.sort()
fwd_musa_dump_tensor_paths = [p for p in musa_dump_tensor_paths if p.startswith("fwd")]
bwd_musa_dump_tensor_paths = [p for p in musa_dump_tensor_paths if p.startswith("bwd")]
fwd_cpu_dump_tensor_paths = [p for p in cpu_dump_tensor_paths if p.startswith("fwd")]
bwd_cpu_dump_tensor_paths = [p for p in cpu_dump_tensor_paths if p.startswith("bwd")]

# forward dump
for musa_tensor_file, cpu_tensor_file in zip(
    fwd_musa_dump_tensor_paths, fwd_cpu_dump_tensor_paths
):
    musa_tensor_path = os.path.join(musa_dump_dir, musa_tensor_file)
    cpu_tensor_path = os.path.join(cpu_dump_dir, cpu_tensor_file)
    musa_tensor = np.load(musa_tensor_path)
    cpu_tensor = np.load(cpu_tensor_path)
    assert np.allclose(
        musa_tensor, cpu_tensor, atol=1e-3
    ), f"Forward tensors differ: {musa_tensor_file} vs {cpu_tensor_file}, max diff={np.max(np.abs(musa_tensor - cpu_tensor))}"
    print(
        f"Forward tensors match successfully: {musa_tensor_file} vs {cpu_tensor_file}"
    )

# backward dump
for musa_tensor_file, cpu_tensor_file in zip(
    bwd_musa_dump_tensor_paths, bwd_cpu_dump_tensor_paths
):
    musa_tensor_path = os.path.join(musa_dump_dir, musa_tensor_file)
    cpu_tensor_path = os.path.join(cpu_dump_dir, cpu_tensor_file)
    musa_tensor = np.load(musa_tensor_path)
    cpu_tensor = np.load(cpu_tensor_path)
    assert np.allclose(
        musa_tensor, cpu_tensor, atol=1e-3
    ), f"Backward tensors differ: {musa_tensor_file} vs {cpu_tensor_file}, max diff={np.max(np.abs(musa_tensor - cpu_tensor))}"
    print(
        f"Backward tensors match successfully: {musa_tensor_file} vs {cpu_tensor_file}"
    )
