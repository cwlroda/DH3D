import os
import argparse
import tensorflow as tf
from tensorpack import *
from tensorpack.predict import PredictConfig
from tensorpack.tfutils.sessinit import SmartInit, SaverRestore
from tensorpack.tfutils.export import ModelExporter

from core.model import DH3D
from core.configs import ConfigFactory

parser = argparse.ArgumentParser(description="Visualise model in Tensorboard")
parser.add_argument("--gpu", help="comma separated list of GPU(s) to use.", default="0")
parser.add_argument("--model", type=str, default="local")
args = parser.parse_args()


def get_nodes(graph_def):
    outputs = [n.name for n in graph_def.node]
    # outputs = set(outputs)

    with open("nodes.txt", "w") as f:
        for output in outputs:
            f.write("Node Name: {}\n".format(output))


def get_vars():
    vars = tf.global_variables()

    with open("vars.txt", "w") as f:
        for var in vars:
            f.write("Variable: {}\n".format(var.name))


def convert_local_model(trained_checkpoint_prefix, export_dir, config):
    pred_config = PredictConfig(
        model=DH3D(config),
        session_init=SmartInit(trained_checkpoint_prefix),
        input_names=[
            "pointclouds",
        ],  # tensor names in the graph, or name of the declared inputs
        output_names=["xyz_feat", "xyz_feat_att"],
    )  # tensor names in the graph

    # session_init = SaverRestore(trained_checkpoint_prefix)
    ModelExporter(pred_config).export_serving(export_dir)


def convert_global_model(trained_checkpoint_prefix, export_dir, config):
    pred_config = PredictConfig(
        model=DH3D(config),
        session_init=SmartInit(trained_checkpoint_prefix),
        input_names=[
            "pointclouds",
        ],  # tensor names in the graph, or name of the declared inputs
        output_names=["globaldesc"],
    )  # tensor names in the graph

    # session_init = SaverRestore(trained_checkpoint_prefix)
    ModelExporter(pred_config).export_serving(export_dir)


def ckpt2sm(trained_checkpoint_prefix, export_dir):
    graph = tf.Graph()

    with tf.Session(graph=graph) as sess:
        # Restore from checkpoint
        loader = tf.train.import_meta_graph(trained_checkpoint_prefix + ".meta")
        loader.restore(sess, trained_checkpoint_prefix)

        graph_def = tf.get_default_graph().as_graph_def()
        get_nodes(graph_def)
        get_vars()

        # prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        #     inputs={
        #         "pointclouds:0": tf.get_default_graph().get_tensor_by_name(
        #             "pointclouds:0"
        #         )
        #     },
        #     outputs={
        #         "xyz_feat:0": tf.get_default_graph().get_tensor_by_name("xyz_feat:0")
        #     },
        #     method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
        # )

        # Export checkpoint to SavedModel
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.SERVING],
            # signature_def_map={
            #     tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
            # },
        )
        builder.save()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    if args.model == "local":
        trained_checkpoint_prefix = "models/local/localmodel"
        export_dir = "models/local/sm"
        cfg = "detection_config"
        configs = ConfigFactory(cfg).getconfig()
        convert_local_model(trained_checkpoint_prefix, export_dir, configs)

    elif args.model == "global":
        trained_checkpoint_prefix = "models/global/globalmodel"
        export_dir = "models/global/sm"
        cfg = "global_config"
        configs = ConfigFactory(cfg).getconfig()
        convert_global_model(trained_checkpoint_prefix, export_dir, configs)

    else:
        print("Invalid model")

    # ckpt2sm(trained_checkpoint_prefix, export_dir)
