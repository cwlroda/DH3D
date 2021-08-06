import os
import sys
import argparse
import tensorflow as tf
from tf_ops.grouping.tf_grouping import group_point
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from tf_ops.interpolation.tf_interpolate import three_nn, three_interpolate

parser = argparse.ArgumentParser(description="Converts model to TensorRT")
parser.add_argument("--model", type=str, default="local", help="Local or global model")
parser.add_argument("--format", type=str, default="")
parser.add_argument(
    "--pb_path", type=str, default="./ckpt/saved_model.pb", help="Path to .pb file",
)
parser.add_argument(
    "--log_dir", type=str, default="./__tb", help="Output log directory",
)
args = parser.parse_args()


def meta2tb(ckpt_path):
    tf.train.import_meta_graph(ckpt_path)
    # for n in tf.get_default_graph().as_graph_def().node:
    #     print(n)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)
        writer.close()


def pb2tb():
    with tf.Session() as sess:
        with gfile.FastGFile(args.pb_path, "rb") as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)

            if len(sm.meta_graphs) != 1:
                print("More than one graph found. Not sure which to write")
                sys.exit(1)

            # graph_def = tf.GraphDef()
            # graph_def.ParseFromString(sm.meta_graphs[0])
            g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)

    train_writer = tf.summary.FileWriter(args.log_dir)
    train_writer.add_graph(sess.graph)
    train_writer.flush()
    train_writer.close()


if __name__ == "__main__":
    if args.format == "pb":
        from tensorflow.python.platform import gfile
        from tensorflow.core.protobuf import saved_model_pb2
        from tensorflow.python.util import compat

        pb2tb()

    elif args.format == "ckpt":
        if args.model == "local":
            ckpt_path = "./models/local/localmodel.meta"
            args.log_dir = os.path.join(args.log_dir, "local")

        elif args.model == "global":
            ckpt_path = "./models/global/globalmodel.meta"
            args.log_dir = os.path.join(args.log_dir, "global")

        meta2tb(ckpt_path)

    else:
        raise ValueError("Unknown format: {}".format(args.format))

    print(
        "Model Imported. Visualize by running: "
        "tensorboard --logdir={}".format(args.log_dir)
    )
