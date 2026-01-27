import json
import sys
import os
import csv
import argparse


class TraceParserV7:
    def __init__(self, json_path):
        self.json_path = json_path
        self.events = []

        # 核心数据结构：External ID -> Group Dictionary
        # Group 结构: { 'aten': [], 'launch': [], 'kernel': [] }
        self.groups = {}

        # 结果集
        self.parsed_records = []

        # 忽略列表 (用于在多个 ATen 算子中选出最有意义的那个)
        self.IGNORE_OPS = {
            "aten::_copy_from",
            "aten::copy_",
            "aten::clone",
            "aten::contiguous",
            "aten::empty",
            "aten::to",
            "aten::_to_copy",
            "aten::detach",
            "aten::alias",
            "aten::resize_",
            "aten::as_strided_",
            "aten::int_repr",
            "aten::select",
            "aten::view",
            "aten::reshape",
            "aten::unsqueeze",
            "aten::squeeze",
            "aten::permute",
            "aten::transpose",
            "aten::expand",
            "aten::repeat",
        }

    def load_trace(self):
        print(f"📂 正在加载: {self.json_path} ...")
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "traceEvents" in data:
                self.events = data["traceEvents"]
            elif isinstance(data, list):
                self.events = data
            else:
                raise ValueError("JSON 格式不正确")

            print(f"✅ 加载成功，共有 {len(self.events)} 个事件。")
            self._process_by_id()

        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback

            traceback.print_exc()

    def _get_arg(self, ev, keys):
        """安全获取 args 中的值"""
        args = ev.get("args", {})
        if not args:
            return None
        # 1. 精确查找
        for k in keys:
            if k in args:
                return args[k]
        # 2. 模糊查找 (忽略大小写)
        for k_args in args.keys():
            k_lower = k_args.lower()
            for kw in keys:
                if kw.lower() in k_lower:
                    return args[k_args]
        return None

    def _process_by_id(self):
        print("⚙️  正在基于 External ID 进行聚类分析...")

        # --- 第一步：归类 (Gathering) ---
        for ev in self.events:
            # 只处理 'X' (Complete) 类型的事件
            if ev.get("ph") != "X":
                continue

            # 获取 External ID
            ext_id = self._get_arg(ev, ["External id", "external_id"])
            if ext_id is None:
                continue

            # 初始化该 ID 的组
            if ext_id not in self.groups:
                self.groups[ext_id] = {"aten": [], "launch": [], "kernel": []}

            # 角色识别逻辑
            name = ev.get("name", "")
            pid = ev.get("pid", -1)

            # 1. GPU Kernel: PID 为 0，或者名字看起来像 Kernel
            #    (加上 name 判断是为了防止某些情况下 pid 不为0的变种)
            is_kernel_name = (
                "musa_asm" in name
                or "musa" in name
                or "kernel" in ev.get("cat", "").lower()
            )
            if pid == 0 or (is_kernel_name and "Launch" not in name):
                self.groups[ext_id]["kernel"].append(ev)

            # 2. Launch Kernel
            elif "Launch" in name:
                self.groups[ext_id]["launch"].append(ev)

            # 3. ATen 算子
            elif name.startswith("aten::"):
                self.groups[ext_id]["aten"].append(ev)

        print(f"📊 共识别出 {len(self.groups)} 个唯一的 External ID 组。正在解析详细信息...")

        # --- 第二步：组内解析 (Linking) ---
        for ext_id, group in self.groups.items():
            # 如果没有 kernel 也没有 launch，可能只是纯 CPU 操作，视需求决定是否导出
            # 这里我们只关心跑在 GPU 上的算子
            if not group["kernel"] and not group["launch"]:
                continue

            # === 1. 挑选最佳 ATen 父节点 ===
            best_aten = None
            aten_candidates = group["aten"]

            if aten_candidates:
                # 策略 A: 优先找不在忽略列表里的
                meaningful_ops = [
                    op for op in aten_candidates if op["name"] not in self.IGNORE_OPS
                ]

                # 策略 B: 在 meaningful 里优先找有 Input Dims 的
                ops_with_shape = [
                    op
                    for op in meaningful_ops
                    if self._get_arg(op, ["Input Dims", "Input shapes", "shapes"])
                ]

                if ops_with_shape:
                    best_aten = ops_with_shape[0]  # 找到完美匹配
                elif meaningful_ops:
                    best_aten = meaningful_ops[0]  # 只有名字没有shape
                else:
                    # 如果全是 copy/contiguous，也没办法，只能选一个最顶层的(通常时间最早的)
                    # 或者选包含 shape 的 copy
                    with_shape = [
                        op
                        for op in aten_candidates
                        if self._get_arg(op, ["Input Dims", "Input shapes"])
                    ]
                    best_aten = with_shape[0] if with_shape else aten_candidates[0]

            # === 2. 挑选 Launch 节点 ===
            # 通常一个 ID 只有一个 Launch，如果有多个，取第一个
            launch_ev = group["launch"][0] if group["launch"] else None

            # === 3. 处理 Kernel (可能是一对多) ===
            # 有时候一个 Launch 会触发多个 Kernel (例如 context 初始化或特殊算子)
            # 或者如果你的 trace 里确实是一对一，这里循环一次就行
            kernels = group["kernel"]

            if not kernels:
                # 只有 Launch 没有 Kernel (可能是 trace 没抓全)
                self._add_record(ext_id, best_aten, launch_ev, None)
            else:
                for k in kernels:
                    self._add_record(ext_id, best_aten, launch_ev, k)

    def _add_record(self, ext_id, parent, launch, kernel):
        # 提取 ATen 信息
        if parent:
            aten_name = parent.get("name")
            shape = self._get_arg(
                parent, ["Input Dims", "Input shapes", "shapes", "dims"]
            )
            stride = self._get_arg(parent, ["Input strides", "strides", "layout"])
            dtype = self._get_arg(parent, ["Input type", "dtype", "input_types"])
        else:
            aten_name = "N/A (No ATen Match)"
            shape = stride = dtype = "N/A"

        # 提取 Launch 信息
        launch_name = launch.get("name") if launch else "N/A"

        # 提取 Kernel 信息
        if kernel:
            k_name = kernel.get("name")
            k_dur = kernel.get("dur", 0) / 1000.0  # ms
            k_args = kernel.get("args", {})

            # Grid/Block 优先查 Kernel，查不到查 Launch
            grid, block = self._get_grid_block(k_args)
            if grid == "N/A" and launch:
                grid, block = self._get_grid_block(launch.get("args", {}))
        else:
            k_name = "N/A"
            k_dur = 0.0
            grid, block = "N/A", "N/A"

        self.parsed_records.append(
            {
                "External ID": ext_id,
                "ATen Parent": aten_name,
                "Launch Node": launch_name,
                "Shape": self._clean_str(shape),
                "Stride": self._clean_str(stride),
                "Dtype": self._clean_str(dtype),
                "GPU Kernel Name": k_name,
                "Duration (ms)": f"{k_dur:.3f}",
                "Grid": grid,
                "Block": block,
            }
        )

    def _clean_str(self, val):
        if val is None:
            return "N/A"
        return str(val).replace("\n", "").replace(" ", "")

    def _get_grid_block(self, args):
        grid = self._get_arg(
            {"args": args}, ["grid", "griddim", "blocks_per_grid", "grid_x"]
        )
        block = self._get_arg(
            {"args": args}, ["block", "blockdim", "threads_per_block", "block_x"]
        )
        return self._clean_str(grid), self._clean_str(block)

    def export_csv(self, output_path):
        if not self.parsed_records:
            print("⚠️ 未解析到任何记录，请检查 JSON 中是否包含 'External id'。")
            return

        # 排序：按 External ID 排序，方便查看
        # 尝试转 int 排序，如果包含字符串则按字符串排
        try:
            self.parsed_records.sort(key=lambda x: int(x["External ID"]))
        except:
            self.parsed_records.sort(key=lambda x: str(x["External ID"]))

        print(f"💾 正在写入 CSV: {output_path} ...")
        headers = list(self.parsed_records[0].keys())

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(self.parsed_records)

        print(f"✅ 完成！已导出 {len(self.parsed_records)} 条 Kernel 记录。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MUSA Trace Parser V7 (External ID Grouping)"
    )
    parser.add_argument("input_file", help="Trace JSON file path")
    parser.add_argument(
        "-o", "--output", default="musa_analysis_v7.csv", help="Output CSV path"
    )

    args = parser.parse_args()

    if os.path.exists(args.input_file):
        parser = TraceParserV7(args.input_file)
        parser.load_trace()
        parser.export_csv(args.output)
    else:
        print("❌ 文件不存在")
