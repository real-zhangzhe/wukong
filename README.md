# 如何 debug
1. 加载冻结的模型权重和固定的输入，并设置随机种子
2. 在模型层之间插入 dump 节点，将 forward / backward 的 tensor 保存成 numpy ndarray
3. 分别运行 cpu / musa tensorflow，分别保存中间层 activation / gradient
4. 离线比较

# 具体执行方式
```bash
# 将编译的得到的 libmusa_plugin.so 替换 exp/debug_tensorflow.py 中的 tf.load_library("/home/albert/Project/tensorflow_musa_extension/build/")
python3 -m exp.debug_tensorflow
```

# 目前报错
```text
muDNN(v3104) 2026-03-12 15:51:06.779594 0d:0h:0m:29s TID=0xe884edaec76228cc GPU=0 Handle=0x7cf0f60 ERROR# NOT_SUPPORTED in MatMul::Run, Reason:
    Unsupported empty tensor                                                                                                                     muDNN(v3104) 2026-03-12 15:51:06.779603 0d:0h:0m:29s TID=0x24932212b5d558c GPU=0 Handle=0x7cf0f60 ERROR# NOT_SUPPORTED in MatMul::Run, Reason:
    Unsupported empty tensor
Traceback (most recent call last):                                      
  File "/root/miniconda3/envs/tf261/lib/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)                                               
  File "/root/miniconda3/envs/tf261/lib/python3.7/runpy.py", line 85, in _run_code                                                 
    exec(code, run_globals)                                                                                                                      
  File "/home/albert/Project/wukong/exp/debug_tensorflow.py", line 248, in <module>
    loss = train_step(inputs, labels)                                                                                                            
  File "/root/miniconda3/envs/tf261/lib/python3.7/site-packages/tensorflow/python/eager/def_function.py", line 885, in __call__            
    result = self._call(*args, **kwds)
  File "/root/miniconda3/envs/tf261/lib/python3.7/site-packages/tensorflow/python/eager/def_function.py", line 950, in _call
    return self._stateless_fn(*args, **kwds)
  File "/root/miniconda3/envs/tf261/lib/python3.7/site-packages/tensorflow/python/eager/function.py", line 3040, in __call__
    filtered_flat_args, captured_inputs=graph_function.captured_inputs)  # pylint: disable=protected-access
  File "/root/miniconda3/envs/tf261/lib/python3.7/site-packages/tensorflow/python/eager/function.py", line 1964, in _call_flat
    ctx, args, cancellation_manager=cancellation_manager))
  File "/root/miniconda3/envs/tf261/lib/python3.7/site-packages/tensorflow/python/eager/function.py", line 596, in call
    ctx=ctx)
  File "/root/miniconda3/envs/tf261/lib/python3.7/site-packages/tensorflow/python/eager/execute.py", line 60, in quick_execute
    inputs, attrs, num_outputs)
tensorflow.python.framework.errors_impl.InternalError: 2 root error(s) found.
  (0) Internal:  MUSA MatMul (2D High Precision) execution failed. Status: 4
         [[node gradient_tape/one_trans/one_trans_block_4/mixed_ffn_4/dense_29/Tensordot/MatMul/MatMul_1 (defined at home/albert/Project/wukong/exp/debug_tensorflow.py:116) ]]
         [[gradient_tape/one_trans/embedding/sparse_embedding/SelectV2_33/_1356]]
  (1) Internal:  MUSA MatMul (2D High Precision) execution failed. Status: 4
         [[node gradient_tape/one_trans/one_trans_block_4/mixed_ffn_4/dense_29/Tensordot/MatMul/MatMul_1 (defined at home/albert/Project/wukong/exp/debug_tensorflow.py:116) ]]
0 successful operations.
0 derived errors ignored. [Op:__inference_train_step_19062]

Errors may have originated from an input operation.
Input Source operations connected to node gradient_tape/one_trans/one_trans_block_4/mixed_ffn_4/dense_29/Tensordot/MatMul/MatMul_1:
 one_trans/one_trans_block_4/mixed_ffn_4/dense_29/Tensordot/Reshape (defined at home/albert/Project/wukong/model/tensorflow/onetrans.py:44)

Input Source operations connected to node gradient_tape/one_trans/one_trans_block_4/mixed_ffn_4/dense_29/Tensordot/MatMul/MatMul_1:
 one_trans/one_trans_block_4/mixed_ffn_4/dense_29/Tensordot/Reshape (defined at home/albert/Project/wukong/model/tensorflow/onetrans.py:44)

Function call stack:
train_step -> train_step
```