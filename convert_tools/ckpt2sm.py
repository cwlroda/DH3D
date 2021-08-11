import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description="Visualise model in Tensorboard")
parser.add_argument("--model", type=str, default="local")
args = parser.parse_args()


def ckpt2sm(trained_checkpoint_prefix, export_dir):
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        # Restore from checkpoint
        loader = tf.compat.v1.train.import_meta_graph(
            trained_checkpoint_prefix + ".meta"
        )
        loader.restore(sess, trained_checkpoint_prefix)

        # Export checkpoint to SavedModel
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.SERVING], strip_default_attrs=True
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
