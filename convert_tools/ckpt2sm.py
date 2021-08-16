import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description="Visualise model in Tensorboard")
parser.add_argument("--model", type=str, default="local")
args = parser.parse_args()


def get_nodes(graph_def):
    node_list = [n.name for n in graph_def.node]
    outputs = set(node_list)

    with open("nodes.txt", "w") as f:
        for output in outputs:
            f.write("Node Name: {}\n".format(output))


def ckpt2sm(trained_checkpoint_prefix, export_dir):
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        # Restore from checkpoint
        loader = tf.compat.v1.train.import_meta_graph(
            trained_checkpoint_prefix + ".meta"
        )
        loader.restore(sess, trained_checkpoint_prefix)

        graph_def = tf.get_default_graph().as_graph_def()
        get_nodes(graph_def)

        # prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        #     inputs={"x_input": tensor_info_x},
        #     outputs={"y_output": tensor_info_y},
        #     method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
        # )

        # Export checkpoint to SavedModel
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.SERVING],
            # signature_def_map={
            #     tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
            # },
        )
        builder.save()


if __name__ == "__main__":
    if args.model == "local":
        trained_checkpoint_prefix = "models/local/localmodel"
        export_dir = "models/local/sm"

    elif args.model == "global":
        trained_checkpoint_prefix = "models/global/globalmodel"
        export_dir = "models/global/sm"

    else:
        print("Invalid model")

    ckpt2sm(trained_checkpoint_prefix, export_dir)
