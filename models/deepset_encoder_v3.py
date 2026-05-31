import tensorflow as tf


class DeepSetPhaseEncoder(tf.keras.Model):
    """
    Deep Sets 编码器：将可变数量的相特征编码为固定维度的置换不变嵌入。

    原理：φ 网络逐相独立映射，ρ 网络对聚合结果再映射。
    结果对相的顺序不敏感（置换不变），对相数无限制。

    关键实现：φ_net 直接作用于 [N, num_phases, F]，Keras Dense
    自动对最后一维操作，等价于对所有 (粒子, 相) 对独立前向传播，
    无需 tf.map_fn 循环，效率更高且 XLA 友好。
    """

    def __init__(self, phi_dims: list, rho_dims: list,
                 aggregation: str = 'mean', name: str = 'DeepSetPhaseEncoder'):
        super().__init__(name=name)

        # φ 网络：逐相特征映射（作用于最后一维，与相数无关）
        self.phi_net = tf.keras.Sequential(name='phi_network')
        for dim in phi_dims:
            self.phi_net.add(tf.keras.layers.Dense(dim, activation='relu'))

        self.aggregation = aggregation

        # ρ 网络：聚合后的全局映射
        self.rho_net = tf.keras.Sequential(name='rho_network')
        for dim in rho_dims:
            self.rho_net.add(tf.keras.layers.Dense(dim, activation='relu'))

    def call(self, inputs: tuple) -> tf.Tensor:
        """
        Args:
            inputs:
                phase_features: [N, num_phases, F]  各相特征，num_phases 可变
                mask          : [N, num_phases]      有效相掩码（bool）
        Returns:
            [N, rho_output_dim]  置换不变的相集合嵌入
        """
        phase_features, mask = inputs

        # φ 网络：Dense 直接对 [N, num_phases, F] 的最后一维操作
        # 等价于对每个 (粒子, 相) 对独立前向传播，无需循环
        # phi_outputs: [N, num_phases, phi_out]
        phi_outputs = self.phi_net(phase_features)

        # 掩码置零（padding 相不贡献）
        mask_expanded = tf.cast(mask[..., tf.newaxis], dtype=phi_outputs.dtype)  # [N, P, 1]
        phi_outputs = phi_outputs * mask_expanded

        # 聚合：沿相维度 reduce
        if self.aggregation == 'sum':
            aggregated = tf.reduce_sum(phi_outputs, axis=1)
        else:  # mean（默认）
            n_valid = tf.maximum(
                tf.reduce_sum(tf.cast(mask, dtype=phi_outputs.dtype), axis=1, keepdims=True), 1.0
            )  # [N, 1]
            aggregated = tf.reduce_sum(phi_outputs, axis=1) / n_valid  # [N, phi_out]

        # ρ 网络：[N, phi_out] → [N, rho_out]
        return self.rho_net(aggregated)
