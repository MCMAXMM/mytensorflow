The TensorFlow Convolution example gives an overview about the difference between SAME and VALID :

For the SAME padding, the output height and width are computed as:
#ceil上界
out_height = ceil(float(in_height) / float(strides[1]))

out_width = ceil(float(in_width) / float(strides[2]))

And

For the VALID padding, the output height and width are computed as:

out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))

out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
