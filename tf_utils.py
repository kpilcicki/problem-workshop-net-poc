import tensorflow as tf

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def get_example_from(data_row):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                    "ax": _float_feature(data_row["ax"]),
                    "ay": _float_feature(data_row["ay"]),
                    "az": _float_feature(data_row["az"]),
                    "gx": _float_feature(data_row["gx"]),
                    "gy": _float_feature(data_row["gy"]),
                    "gz": _float_feature(data_row["gz"])
                }
            )
        )