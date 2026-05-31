import tensorflow as tf
try:
    from utils.convolutions import ContinuousConv
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from convolutions import ContinuousConv
import open3d.ml.tf as ml3d
import numpy as np  # 导入 NumPy，虽然在这个简化示例中没有直接使用，但通常用于数组操作

class MultiDensityContinuousConv(ContinuousConv):
    """
    自定义的连续卷积层，用于处理多密度流体。
    继承自 open3d.ml.tf.layers.ContinuousConv，并重写 call 方法以实现
    基于密度加权的邻居重要性计算。
    """
    def __init__(self,
                 filters,  # 输出特征的通道数
                 kernel_size,  # 卷积核的空间分辨率，例如 [3, 3, 3]
                 activation=None,  # 激活函数
                 use_bias=True,  # 是否使用偏置项
                 kernel_initializer='uniform',  # 卷积核权重的初始化器
                 bias_initializer='zeros',  # 偏置项的初始化器
                 kernel_regularizer=None,  # 卷积核权重的正则化器
                 bias_regularizer=None,  # 偏置项的正则化器
                 align_corners=True,  # 坐标映射时是否对齐角点
                 coordinate_mapping='ball_to_cube_radial',  # 坐标映射方式
                 interpolation='linear',  # 插值方式
                 normalize=True,  # 是否进行归一化
                 radius_search_ignore_query_points=False,  # 半径搜索时是否忽略查询点自身
                 radius_search_metric='L2',  # 半径搜索的距离度量标准
                 offset=None,  # 偏移量
                 # window_function 应该在这里设置为 None，或者以其他方式处理。
                 # 我们将手动计算邻居重要性。
                 window_function=None,  # 用于空间加权的窗口函数 (会被我们自定义的取代)
                 combined_importance_function=None,
                 use_dense_layer_for_center=False,  # 是否使用密集层处理中心点
                 dense_kernel_initializer='glorot_uniform',  # 密集层权重的初始化器
                 dense_kernel_regularizer=None,  # 密集层权重的正则化器
                 symmetric=False,
                 sym_axis=2,
                 circular=False,
                 **kwargs):
        """
        初始化 MultiDensityContinuousConv 层。
        注意：将 window_function=None 传递给基类，以便它不计算默认的重要性。
        """
        # 调用父类的 __init__ 方法，传入所有参数，确保基类正确初始化
        super().__init__(filters=filters, kernel_size=kernel_size,
                         activation=activation, use_bias=use_bias,
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         align_corners=align_corners, coordinate_mapping=coordinate_mapping,
                         interpolation=interpolation, normalize=normalize,
                         radius_search_ignore_query_points=radius_search_ignore_query_points,
                         radius_search_metric=radius_search_metric, offset=offset,
                         window_function=combined_importance_function,  # 确保它被正确处理 (通常为 None), 这里传递是为了正确初始化self.fixed_radius_search
                         use_dense_layer_for_center=use_dense_layer_for_center,
                         dense_kernel_initializer=dense_kernel_initializer,
                         dense_kernel_regularizer=dense_kernel_regularizer,
                         symmetric=symmetric, sym_axis=sym_axis, circular=circular,
                         **kwargs)

        # 存储窗口函数，如果提供了的话，用于自定义重要性计算
        self._density_window_function = combined_importance_function  # 用于基于密度进行窗口化的自定义参数
        # 你可能需要存储一个单独的函数用于密度加权

    def call(self,
             inp_features,  # 输入特征，形状为 [num_input_points, in_channels]
             inp_positions,  # 输入点坐标，形状为 [num_input_points, 3]
             out_positions,  # 输出点坐标，形状为 [num_output_points, 3]
             extents,  # 扩展范围，形状为 [1] 或 [num_output_points]，定义了邻域搜索的半径
             # 添加密度输入
             inp_densities,  # 输入点密度，形状为 [num_input_points, 1]
             out_densities,  # 输出点密度，形状为 [num_output_points, 1]
             inp_importance=None,  # 可选的输入点重要性
             fixed_radius_search_hash_table=None,  # 可选的固定半径搜索哈希表
             # 我们将自己计算并传递这些参数
             user_neighbors_index=None,  # 必须为 None 才能使用此层的自定义邻居搜索逻辑
             user_neighbors_row_splits=None,  # 必须为 None 才能使用此层的自定义邻居搜索逻辑
             user_neighbors_importance=None,  # 必须为 None 才能使用此层的自定义邻居搜索逻辑
             ):
        """
        计算多密度连续卷积。

        inp_features: 输入特征，每个输入点都有一个特征向量。
        inp_positions: 输入点的位置坐标。
        out_positions: 输出点的位置坐标，也就是计算卷积的位置。
        extents: 定义了每个输出点周围邻域搜索的范围。
        inp_densities: 每个输入点的密度值.
        out_densities: 每个输出点的密度值.
        inp_importance: (可选) 输入点的重要性值。
        fixed_radius_search_hash_table: (可选) 预先计算的用于加速邻域搜索的哈希表。
        """

        # 确保 user_neighbors_* 没有被预先提供，以强制使用此层的邻居搜索逻辑
        if user_neighbors_index is not None or user_neighbors_row_splits is not None or user_neighbors_importance is not None:
            raise ValueError("MultiDensityContinuousConv 使用内部邻居搜索计算，不要提供 user_neighbors_* 参数")

        # --- 1. 执行邻域搜索 ---
        # 使用在基类 __init__ 方法中已经初始化的半径搜索层
        return_distances = True  # 需要距离以进行潜在的空间加权
        if extents.shape.rank == 0:  # 所有输出点使用相同的范围
            radius = 0.5 * extents
            self.nns = self.fixed_radius_search(
                inp_positions,
                queries=out_positions,
                radius=radius,
                hash_table=fixed_radius_search_hash_table
                # return_distances=return_distances  # 确保返回距离
            )
            # 归一化距离，以便用于窗口函数（如果使用 L2 距离）
            if return_distances and self.radius_search_metric == 'L2':
                neighbors_distance_normalized = self.nns.neighbors_distance / (radius * radius)
            elif return_distances and self.radius_search_metric == 'L1': # 修正：针对L1的归一化
                neighbors_distance_normalized = self.nns.neighbors_distance / radius
            elif return_distances:  # Linf - 通常不以这种方式进行归一化，可能需要自定义归一化
                neighbors_distance_normalized = self.nns.neighbors_distance  # 或者定义自定义归一化
            else:
                neighbors_distance_normalized = None # 没有距离


        elif extents.shape.rank == 1:  # 每个输出点都有不同的范围
            radii = 0.5 * extents
            # self.nns = self.radius_search(inp_positions,
            #                           queries=out_positions,
            #                           radii=radii,
            #                           return_distances=return_distances,  # 确保返回距离
            #                           normalize_distances=return_distances)  # 基类半径搜索为 window_function 归一化距离
            self.nns = self.radius_search(inp_positions,
                                      queries=out_positions,
                                      radii=radii)
            if return_distances:
                neighbors_distance_normalized = self.nns.neighbors_distance_normalized
            else:
                neighbors_distance_normalized = None


        else:
            raise Exception("extents 的秩必须为 0 或 1")

        # --- 2. 提取邻居点和查询点的密度信息 ---
        neighbors_index = self.nns.neighbors_index  # 邻居索引，指示哪些输入点是邻居
        neighbors_row_splits = self.nns.neighbors_row_splits  # 行分割，指示每个输出点的邻居列表的起始和结束位置

        # 收集 *邻居* 点的密度
        neighbor_densities = tf.gather(inp_densities, neighbors_index)  # 形状 [num_pairs, 1]

        # 收集与每个邻居对对应的 *查询* 点的密度
        # point_idx 张量将每个邻居映射回其查询点索引
        # query_point_idx_for_neighbors = ml3d.ops.row_splits_to_point_idx(             ### 当前版本不存在这个函数,进行手动计算
        #     neighbors_row_splits)  # 形状 [num_pairs] - 将 row splits 转换为点索引
        # 替代方案：手动计算每个邻居对应的查询点索引
        # 1. 获取每个查询点的邻居数量
        # neighbors_row_splits 的形状是 [num_output_points + 1]
        # 其中 neighbors_row_splits[i+1] - neighbors_row_splits[i] 是查询点 i 的邻居数量
        num_neighbors_per_query = neighbors_row_splits[1:] - neighbors_row_splits[:-1] # 形状 [num_output_points]

        # 2. 获取查询点的索引列表
        query_indices = tf.range(tf.shape(out_positions)[0], dtype=neighbors_row_splits.dtype) # 形状 [num_output_points]

        # 3. 重复每个查询点索引，重复次数为其对应的邻居数量
        query_point_idx_for_neighbors = tf.repeat(query_indices, num_neighbors_per_query) # 形状 [num_pairs]

        # 现在 query_point_idx_for_neighbors 的形状和内容与 ml3d.ops.row_splits_to_point_idx 应该相同
        # 接下来使用这个张量来收集查询点密度
        query_densities_for_neighbors = tf.gather(out_densities,
                                                  query_point_idx_for_neighbors)  # 形状 [num_pairs, 1]

        # --- 3. 计算自定义邻居重要性 ---
        # 首先考虑空间加权 # 形状 [num_pairs, 1] 或 [num_pairs]
        if self._density_window_function is not None and neighbors_distance_normalized is not None:
            custom_neighbors_importance = self._density_window_function(neighbors_distance_normalized, neighbor_densities, query_densities_for_neighbors)  # 应用空间窗口函数
        elif self.window_function is not None and neighbors_distance_normalized is not None:
            # 如果提供了，则回退到基类窗口函数
            custom_neighbors_importance = self.window_function(neighbors_distance_normalized)
        else:
            custom_neighbors_importance = tf.ones_like(neighbor_densities)  # 没有空间加权，默认为 1

        # 对于单通道重要性，如果需要，通过挤压确保它是秩 1
        if custom_neighbors_importance.shape.rank == 2 and custom_neighbors_importance.shape[-1] == 1:
            custom_neighbors_importance = tf.squeeze(custom_neighbors_importance, axis=-1)  # 形状 [num_pairs]

        # 处理 inp_importance，如果使用的话 - 通常在聚集之前应用于输入特征 *之前*
        # 如果 inp_importance 是逐点的，则聚集它以用于邻居可能是：
        # if inp_importance is not None:
        #    neighbor_inp_importance = tf.gather(inp_importance, neighbors_index)
        #    custom_neighbors_importance *= neighbor_inp_importance  # 与输入重要性组合

        # --- 4. 调用基类 ContinuousConv 逻辑，使用自定义邻居/重要性 ---
        # 将计算的邻居和重要性传递给父类 call 方法
        # 传递原始 inp_features、inp_positions、out_positions、extents
        # 设置 user_neighbors_* 参数
        out_features = super().call(
            inp_features=inp_features,
            inp_positions=inp_positions,
            out_positions=out_positions,
            extents=extents,
            inp_importance=inp_importance,  # 传递原始 inp_importance
            fixed_radius_search_hash_table=fixed_radius_search_hash_table,
            user_neighbors_index=neighbors_index,  # 使用我们计算的邻居
            user_neighbors_row_splits=neighbors_row_splits,  # 使用我们计算的行分割
            user_neighbors_importance=custom_neighbors_importance,  # 使用我们计算的重要性
            # 其他 kwargs 会在使用签名/调用中的 **kwargs 时隐式传递
        )

        # 激活和偏置在基类 call 方法的返回中添加

        return out_features

    # 你可能需要调整 build，如果 inp_features 不再包含密度通道
    # 或者显式传递 in_channels，如果从特征中移除密度
    def build(self, inp_features_shape):
        # 如果从 inp_features 中移除密度，则在此处调整 in_channels
        # self.in_channels = inp_features_shape[-1] # 默认
        # 如果密度是第一个通道并被移除：
        # self.in_channels = inp_features_shape[-1] - 1 # 或者无论密度通道计数是什么

        # 如果密度保持在 inp_features 中，则此处无需更改 self.in_channels

        super().build(inp_features_shape)  # 调用基类 build 以创建内核/偏置权重


# 如果你想应用距离衰减,  可以将一个距离衰减函数传递给 window_function
def spatial_window_fn(r_sqr):
    """示例：使用 (1 - r^2)^3 形式的窗口函数"""
    return tf.clip_by_value((1 - r_sqr)**3, 0, 1)

# 如果你要密度也参与空间衰减，设置`density_window_function`，或者完全自定义组合
def relative_density_importance(normalized_distance, neighbor_relative_densities, query_relative_densities):
    """
    计算结合了距离和相对密度的邻居重要性。

    Args:
        normalized_distance: 归一化距离张量 [num_pairs]
        neighbor_relative_densities: 邻居点相对密度张量 [num_pairs, 1]
        query_relative_densities: 查询点相对密度张量 [num_pairs, 1]

    Returns:
        组合重要性张量 [num_pairs]
    """
    # 空间衰减部分
    spatial_weight = tf.clip_by_value((1 - normalized_distance)**3, 0, 1)

    # 使用相对密度作为密度权重
    # 可以直接使用相对邻居密度
    # density_weight = tf.squeeze(neighbor_relative_densities, axis=-1) # 形状 [num_pairs]

    # 也可以是相对密度比（邻居相对密度 / 查询点相对密度）
    # 注意处理查询点相对密度接近零的情况（如果查询点密度接近零）
    # 如果查询点密度是参考密度的倍数，并且参考密度远大于实际可能的最小密度，
    # 那么 query_relative_densities 可能会接近零，需要加 epsilon
    epsilon = 1e-6
    density_ratio_weight = tf.squeeze(neighbor_relative_densities, axis=-1) / (tf.squeeze(query_relative_densities, axis=-1) + epsilon)
    print(neighbor_relative_densities.shape, query_relative_densities.shape)
    # 可以直接使用相对邻居密度的乘积
    # density_weight = tf.squeeze(neighbor_relative_densities, axis=-1) * tf.squeeze(query_relative_densities, axis=-1)
    # 你可以选择 density_weight = density_ratio_weight
    density_weight = density_ratio_weight

    # 组合权重
    combined_weight = spatial_weight * density_weight # 或者 spatial_weight * density_ratio_weight

    return combined_weight

if __name__ == "__main__":

    # --- 示例用法 ---
    # 1. 创建一些示例数据
    num_input_points = 100
    num_output_points = 50
    in_channels = 4  # 除密度外的特征数量
    inp_features = tf.random.normal((num_input_points, in_channels)) # 输入特征，注意这里暂时不包含密度
    inp_positions = tf.random.normal((num_input_points, 3))  # 输入点坐标
    out_positions = tf.random.normal((num_output_points, 3))  # 输出点坐标
    extents = tf.constant(2.0, dtype=tf.float32)  # 定义邻域搜索范围
    # 假设密度为 0 到 1 之间的值
    inp_densities = tf.random.uniform((num_input_points, 1), minval=0.5, maxval=10.0, dtype=tf.float32) # 输入点密度，形状为 [num_input_points, 1]
    out_densities = tf.random.uniform((num_output_points, 1), minval=0.5, maxval=10.0, dtype=tf.float32) # 输出点密度，形状为 [num_output_points, 1]

    # 2. 创建 MultiDensityContinuousConv 层实例

    conv = MultiDensityContinuousConv(
        filters=16,  # 输出通道数
        kernel_size=[4, 4, 4],  # 卷积核大小
        coordinate_mapping='ball_to_cube_radial',  # 坐标映射方式
        interpolation='linear',  # 插值方式
        normalize=True,  # 是否进行归一化
        #  要覆盖原始 window_function的逻辑,设置为None
        window_function=None, #  添加距离衰减 (可以设置为None, 如果所有衰减都在自定义 call 中处理)
        combined_importance_function=relative_density_importance
        # 传入了 density_window_function
        #density_window_function=density_window_fn, # 如果需要基于距离和密度的更复杂的窗口化
        # 调整 build if you remove density channel from feature.
        # in_channels must the same number of channels you are using in the kernel
    )

    print(type(conv).__name__)  # 输出: MultiDensityContinuousConv

    # 构建图层, 密度不包含在特征之中，所以图层可以正常运行
    conv.build(inp_features.shape)

    # 3. 调用该层
    # 假设 densities 已知，并作为单独的张量传递
    # 为了便于说明，假设 out_densities 与 inp_densities 相同

    out_features = conv(
        inp_features=inp_features,
        inp_positions=inp_positions,
        out_positions=out_positions,
        extents=extents,
        inp_densities=inp_densities,  # 传入输入密度
        out_densities=out_densities,  # 传入输出密度（如果与输入位置不同）
    )

    # 4. 检查结果
    print("Output features shape:", out_features.shape)  # 预期输出：(num_output_points, 16)
    num_neighbors = ml3d.ops.reduce_subarrays_sum(
            tf.ones_like(conv.nns.neighbors_index,
                         dtype=tf.float32),
            conv.nns.neighbors_row_splits)
    print("Number of neighbors:", num_neighbors)