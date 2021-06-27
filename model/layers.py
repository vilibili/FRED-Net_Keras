from keras import backend as K
from keras.layers import Layer
import numpy as np


class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == "tensorflow":
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(
                inputs, ksize=ksize, strides=strides, padding=padding
            )
        else:
            errmsg = "{} backend is not supported for layer {}".format(
                K.backend(), type(self).__name__
            )
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, "int32")
            input_shape = K.tf.shape(updates, out_type="int32")
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.size[0],
                    input_shape[2] * self.size[1],
                    input_shape[3],
                )
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype="int32")
            batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = K.reshape(
                K.tf.range(output_shape[0], dtype="int32"), shape=batch_shape
            )
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype="int32")
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3],
        )

def unpool(pool, ind, ksize=[1, 2, 2, 1], name=None):
    with K.tf.variable_scope('name') as scope:
        input_shape = K.tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

        flat_input_size = K.tf.cumprod(input_shape)[-1]

        flat_output_shape = K.tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

        pool_ = K.tf.reshape(pool, K.tf.stack([flat_input_size]))
        batch_range = K.tf.reshape(K.tf.range(K.tf.cast(output_shape[0], K.tf.int64), dtype=ind.dtype),
                                 shape=K.tf.stack([input_shape[0], 1, 1, 1]))
        b = K.tf.ones_like(ind) * batch_range
        b = K.tf.reshape(b, K.tf.stack([flat_input_size, 1]))
        ind_ = K.tf.reshape(ind, K.tf.stack([flat_input_size, 1]))
        ind_ = ind_ - b * K.tf.cast(flat_output_shape[1], K.tf.int64)
        ind_ = K.tf.concat([b, ind_], 1)

        ret = K.tf.scatter_nd(ind_, pool_, shape=K.tf.cast(flat_output_shape, K.tf.int64))
        ret = K.tf.reshape(ret, K.tf.stack(output_shape))

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * ksize[1], set_input_shape[2] * ksize[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)
    return ret

def unpool_with_argmax(pool, ind, name = None, ksize=[1, 2, 2, 1]):

    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    input_shape = pool.get_shape().as_list()
    print(input_shape)

    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    print(output_shape)

    flat_input_size = np.prod([2, input_shape[1], input_shape[2], input_shape[3]])
    print(flat_input_size)

    flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

    pool_ = K.tf.reshape(pool, [flat_input_size])
    batch_range = K.tf.reshape(K.tf.range(2, dtype=ind.dtype), shape=[2, 1, 1, 1])
    b = K.tf.ones_like(ind) * batch_range
    b = K.tf.reshape(b, [flat_input_size, 1])
    ind_ = K.tf.reshape(ind, [flat_input_size, 1])
    ind_ = K.tf.concat([b, ind_], 1)

    ret = K.tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
    ret = K.tf.reshape(ret, output_shape)
    return ret