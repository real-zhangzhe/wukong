# 如何 debug
1. 加载冻结的模型权重和固定的输入，并设置随机种子
2. 在模型层之间插入 dump 节点，将 forward / backward 的 tensor 保存成 numpy ndarray
3. 分别运行 cpu / musa tensorflow，分别保存中间层 activation / gradient
4. 离线比较

# 具体执行方式
```bash
# 跑 musa tensorflow，dump 中间结果（DUMP_DIR 环境变量表示 dump 目录）
DUMP_DIR="./debug_dumps_musa" python3 -m exp.debug_tensorflow

# 注释 exp/debug_tensorflow.py 的 tf.load_library("/home/albert/Project/tensorflow_musa_extension/build/libmusa_plugin.so")，回退到 cpu tensorflow
# 跑 cpu tensorflow，dump 中间结果
DUMP_DIR="./debug_dumps_cpu" python3 -m exp.debug_tensorflow

# 离线对比
python3 -m exp.find_diff
```

# 目前的对比结果
1. 在设置 DROPOUT > 0.0 时，forward 会在 dropout 层开始对不上
2. 当设置 DROPOUT = 0.0 时，forward 能对上，backward 层从 batchnorm 层开始对不上