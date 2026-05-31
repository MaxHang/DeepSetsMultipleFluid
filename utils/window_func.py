import tensorflow as tf
from debug_utils import debug_print


def get_window_func(typ, fac=1.0, **kwargs):
    if typ == "poly6":

        def func(q):
            return fac * tf.clip_by_value((1 - q)**3, 0, 1)
    elif typ == "cubic":

        def func(q):
            q_sqrt = tf.sqrt(q)
            return fac * 4 / 3 * tf.compat.v1.where(
                q <= 1,
                tf.compat.v1.where(q_sqrt <= 0.5, 6 * (q_sqrt**3 - q) + 1, 2 *
                                   (1 - q_sqrt)**3), tf.zeros_like(q_sqrt))
    elif typ == "linear":

        def func(q):
            q_sqrt = tf.sqrt(q)
            return fac * (1 - q_sqrt)
    elif typ == "peak":

        def func(q):
            q_sqrt = tf.sqrt(q)
            return fac * (1 - 2 * q_sqrt + q)
    elif typ == "cubic_grad":

        def func(q):
            # return tf.where(q <= 1, 1.0, 0.0)
            q_sqrt = tf.sqrt(q)
            return fac * 4 / 3 * tf.compat.v1.where(
                q <= 1,
                tf.compat.v1.where(q_sqrt <= 0.5, 18 * q - 12 * q_sqrt, -6 *
                                   (1 - q_sqrt)**2), tf.zeros_like(q_sqrt))
    elif typ == "custom_density_window":
        def func(normalized_distance, neighbor_relative_densities, query_relative_densities):
            # 空间衰减部分
            spatial_weight = tf.clip_by_value(
                (1 - normalized_distance)**3, 0, 1)
            # 可以直接使用相对邻居密度的乘积
            # density_weight = tf.squeeze(neighbor_relative_densities, axis=-1) * tf.squeeze(query_relative_densities, axis=-1)
            # 也可以是相对密度比（邻居相对密度 / 查询点相对密度）
            # 注意处理查询点相对密度接近零的情况（如果查询点密度接近零）
            epsilon = 1e-6
            density_ratio_weight = tf.squeeze(neighbor_relative_densities, axis=-1) / (
                tf.squeeze(query_relative_densities, axis=-1) + epsilon)
            debug_print("density_ratio_weight max, min: ", tf.reduce_max(density_ratio_weight), tf.reduce_min(density_ratio_weight))
            density_weight = density_ratio_weight
            combined_weight = spatial_weight * density_weight
            return fac * combined_weight

    elif typ == "custom_density_window_sym":
        def func(normalized_distance, neighbor_relative_densities, query_relative_densities):
            # 空间衰减部分
            q_sqrt = tf.sqrt(normalized_distance)
            spatial_weight = 1 - 2 * q_sqrt + normalized_distance
            # 可以直接使用相对邻居密度的乘积
            # density_weight = tf.squeeze(neighbor_relative_densities, axis=-1) * tf.squeeze(query_relative_densities, axis=-1)
            # 也可以是相对密度比（邻居相对密度 / 查询点相对密度）
            # 注意处理查询点相对密度接近零的情况（如果查询点密度接近零）
            epsilon = 1e-6
            density_ratio_weight = tf.squeeze(neighbor_relative_densities, axis=-1) / (
                tf.squeeze(query_relative_densities, axis=-1) + epsilon)
            # debug_print("density_ratio_weight step 3000: ", density_ratio_weight[::3000])
            # debug_print("density_ratio_weight.shape: ", density_ratio_weight.shape)
            debug_print("density_ratio_weight max, min: ", tf.reduce_max(density_ratio_weight), tf.reduce_min(density_ratio_weight))
            density_weight = density_ratio_weight
            # debug_print("spatial_weight step 3000: ", spatial_weight[::3000])
            # debug_print("spatial_weight.shape: ", spatial_weight.shape)
            combined_weight = spatial_weight * density_weight
            # debug_print("combined_weight step 3000: ", combined_weight[::3000])
            # debug_print("combined_weight.shape: ", combined_weight.shape)
            return fac * combined_weight

    elif typ is None:
        func = None
    else:
        raise NotImplementedError()
    return func
