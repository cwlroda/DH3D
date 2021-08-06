import os
import sys
import argparse
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tf_ops.grouping.tf_grouping import group_point
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from tf_ops.interpolation.tf_interpolate import three_nn, three_interpolate

parser = argparse.ArgumentParser(description="Converts model to TensorRT")
parser.add_argument("--model", type=str, default="local", help="Local or global model")
parser.add_argument(
    "--format", type=str, default="ckpt2trt", help="Conversion from .ckpt or .pb"
)
parser.add_argument(
    "--log_dir", type=str, default="./trt", help="Output log directory",
)
args = parser.parse_args()

INPUTS = ["pointclouds:0", "knn_inds:0"]
OUTPUTS = ["feat_l2normed", "xyz_feat:0"]


def ckpt2trt():
    # First create a `Saver` object (for saving and rebuilding a
    # model) and import your `MetaGraphDef` protocol buffer into it:
    saver = tf.train.import_meta_graph(args.meta_path)

    # Then restore your training data from checkpoint files:
    saver.restore(sess, args.ckpt_path)

    # Finally, freeze the graph:
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        output_node_names=OUTPUTS,
        variable_names_blacklist=[],
    )

    tf.train.write_graph(
        frozen_graph, "./export_dir/0/", "saved_model.pb", as_text=False
    )

    converter = trt.TrtGraphConverter(
        input_graph_def=frozen_graph, nodes_blacklist=OUTPUTS
    )
    trt_graph = converter.convert()

    # saved_model_dir_trt = os.path.join(args.log_dir, "tensorrt_model.trt")
    # converter.save(saved_model_dir_trt)

    return trt_graph


if __name__ == "__main__":
    with tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
    ) as sess:
        if args.format == "ckpt2trt":
            if args.model == "local":
                ckpt_path = "./models/local/localmodel.meta"
                args.log_dir = os.path.join(args.log_dir, "local")

            elif args.model == "global":
                ckpt_path = "./models/global/globalmodel.meta"
                args.log_dir = os.path.join(args.log_dir, "global")

            trt_graph = ckpt2trt()

        # Import the TensorRT graph into a new graph and run:
        # g = tf.get_default_graph().as_graph_def()
        # TODO: get input tensor name
        # inputs = [g.get_tensor_by_name(name) for name in inputs]
        # print(inputs)
        # TODO: map inputs
        # output_node = tf.import_graph_def(trt_graph, input_map={}, return_elements=outputs)
        # print("Output node: ")
        # print(output_node)
        # TODO: add input node to feed_dict
        # sess.run(output_node, feed_dict={})
