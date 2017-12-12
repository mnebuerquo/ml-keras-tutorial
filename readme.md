# Kera Tutorial

This is from [an online tutorial](https://elitedatascience.com/keras-tutorial-deep-learning-in-python).

## To Run the Program

In the project directory:
```bash
./run keras-tut.py
```

## Exception
```
keras-tut.py:49: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (3, 3), input_shape=(1, 28, 28..., activation="relu")`
  model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
Traceback (most recent call last):
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/tensorflow/python/framework/common_shapes.py", line 686, in _call_cpp_shape_fn_impl
    input_tensors_as_shapes, status)
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/tensorflow/python/framework/errors_impl.py", line 473, in __exit__
    c_api.TF_GetCode(self.status.status))
tensorflow.python.framework.errors_impl.InvalidArgumentError: Negative dimension size caused by subtracting 3 from 1 for 'conv2d_1/convolution' (op: 'Conv2D') with input shapes: [?,1,28,28], [3,3,28,32].

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "keras-tut.py", line 49, in <module>
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/keras/models.py", line 464, in add
    layer(x)
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/keras/engine/topology.py", line 603, in __call__
    output = self.call(inputs, **kwargs)
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/keras/layers/convolutional.py", line 164, in call
    dilation_rate=self.dilation_rate)
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py", line 3195, in conv2d
    data_format=tf_data_format)
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/tensorflow/python/ops/nn_ops.py", line 751, in convolution
    return op(input, filter)
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/tensorflow/python/ops/nn_ops.py", line 835, in __call__
    return self.conv_op(inp, filter)
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/tensorflow/python/ops/nn_ops.py", line 499, in __call__
    return self.call(inp, filter)
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/tensorflow/python/ops/nn_ops.py", line 187, in __call__
    name=self.name)
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/tensorflow/python/ops/gen_nn_ops.py", line 631, in conv2d
    data_format=data_format, name=name)
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/tensorflow/python/framework/op_def_library.py", line 787, in _apply_op_helper
    op_def=op_def)
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 2958, in create_op
    set_shapes_for_outputs(ret)
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 2209, in set_shapes_for_outputs
    shapes = shape_func(op)
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 2159, in call_with_requiring
    return call_cpp_shape_fn(op, require_shape_fn=True)
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/tensorflow/python/framework/common_shapes.py", line 627, in call_cpp_shape_fn
    require_shape_fn)
  File "/Users/sa35907/projects/deepmind/keras-tutorial/env/lib/python3.5/site-packages/tensorflow/python/framework/common_shapes.py", line 691, in _call_cpp_shape_fn_impl
    raise ValueError(err.message)
ValueError: Negative dimension size caused by subtracting 3 from 1 for 'conv2d_1/convolution' (op: 'Conv2D') with input shapes: [?,1,28,28], [3,3,28,32].
```
