#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多相流体模拟网络

核心设计：
  1. 位置分支：DeepSet 编码相集合（vf + density）→ 置换不变的全局相嵌入，
       只服务于位置/速度预测，不依赖相数。
  2. VF 分支：独立的 density-free DeepSet 编码当前 VF 集合，
       再结合可选的速度和逐相 vf_i 构建空间上下文，并在最终逐相预测时注入 cd/cf 条件，
       避免密度与 cd/cf 在 VF 分支中相互污染。
  3. cd/cf 条件编码：MLP 将 (cd, cf) 编码为向量，仅注入 VF 分支的最终逐相决策层。

去掉了 max_num_phases 和动态 VF 分支头——两者均依赖"预先知道相数"，
与 DeepSet 相数无关的设计原则矛盾。
"""

import tensorflow as tf
import open3d.ml.tf as ml3d
import numpy as np
from typing import Tuple, List, Optional
from models.deepset_encoder_v3 import DeepSetPhaseEncoder


class MultiPhaseParticleNetwork(tf.keras.Model):
    """
    数据驱动多相流体模拟网络。

    位置预测：DeepSet(vf+density) + ContinuousConv 主干 → 位置修正头
    VF  预测：density-free VF 分支 + 逐相特征 → 按相共享 CConv → delta_vf/绝对 VF → 守恒归一化

    关键设计：VF 逐相预测器的权重在所有相之间共享，
    因此训练时用 2 相，推理时可直接泛化到任意相数，无需 max_num_phases。
    """

    def __init__(self,
                 # 网络结构
                 kernel_size: List[int] = [4, 4, 4],
                 layer_channels: List[int] = [32, 64, 64],
                 # 相特征编码
                 phase_feat_centralization: bool = True,
                 aggregation: str = 'mean',
                 # 条件参数
                 cd_cf_as_input: bool = True,
                 cd_cf_embedding_dim: int = 32,
                 # 物理/仿真参数
                 particle_radius: float = 0.05,
                 radius_scale: float = 1.5,
                 timestep: float = 1 / 50,
                 gravity: Tuple[float, float, float] = (0, -9.81, 0),
                 # 卷积参数
                 coordinate_mapping: str = 'ball_to_cube_volume_preserving',
                 interpolation: str = 'linear',
                 use_window: bool = True,
                 # VF 预测模式
                 # False（默认）= 直接预测：网络输出未归一化 VF，再经 ReLU + 归一化约束
                 # True = 残差预测：网络输出 delta_vf，current_vf + alpha * delta_vf 后再归一化
                 vf_residual: bool = False,
                 alpha: float = 0.1,
                 vf_use_velocity: bool = False,
                 ) -> None:
        super().__init__(name=type(self).__name__)

        init_vars = locals()
        self.init_params = {k: v for k, v in init_vars.items() if k != 'self'}

        self.layer_channels = layer_channels
        self.phase_feat_centralization = phase_feat_centralization
        self.aggregation = aggregation
        self.cd_cf_as_input = cd_cf_as_input
        self.cd_cf_embedding_dim = cd_cf_embedding_dim
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.timestep = timestep
        self.gravity = tf.constant(gravity, dtype=tf.float32)
        self.filter_extent = np.float32(radius_scale * 6 * particle_radius)
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.vf_residual = vf_residual  # False=直接预测, True=delta_vf 残差预测
        self.alpha = alpha              # Only used when vf_residual=True
        self.vf_use_velocity = vf_use_velocity

        self._all_convs = []

        def window_poly6(r_sqr):
            return tf.clip_by_value((1 - r_sqr) ** 3, 0, 1)

        def Conv(name, filters, activation=None, **kwargs):
            window_fn = window_poly6 if self.use_window else None
            radius_search_ignore_query_points = kwargs.pop(
                'radius_search_ignore_query_points', True
            )
            conv = ml3d.layers.ContinuousConv(
                name=name,
                filters=filters,
                kernel_size=self.kernel_size,
                activation=activation,
                align_corners=True,
                interpolation=self.interpolation,
                coordinate_mapping=self.coordinate_mapping,
                normalize=False,
                window_function=window_fn,
                radius_search_ignore_query_points=radius_search_ignore_query_points,
                **kwargs,
            )
            self._all_convs.append((name, conv))
            return conv

        # ── DeepSet 相编码器 ────────────────────────────────────────────────
        # 输入: [N, num_phases, 2]  输出: [N, 64]
        # 对相集合置换不变，相数可变
        self.phase_encoder = DeepSetPhaseEncoder(
            phi_dims=[64, 128],
            rho_dims=[128, 64],
            aggregation=self.aggregation,
            name='phase_encoder'
        )

        # ── VF 分支专用 DeepSet 编码器（仅看 VF，不看密度）──────────────────────
        # 输入: [N, num_phases, 1]  输出: [N, 64]
        # 仅为 VF 更新构建 density-free 的置换不变上下文
        self.vf_phase_encoder = DeepSetPhaseEncoder(
            phi_dims=[64, 128],
            rho_dims=[128, 64],
            aggregation=self.aggregation,
            name='vf_phase_encoder'
        )

        # ── cd/cf 条件编码器 ────────────────────────────────────────────────
        if self.cd_cf_as_input:
            self.cd_cf_encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', name='cd_cf_enc1'),
                tf.keras.layers.Dense(cd_cf_embedding_dim, activation='relu', name='cd_cf_enc2'),
            ], name='cd_cf_encoder')
            _ = self.cd_cf_encoder(tf.zeros((1, 2), dtype=tf.float32))

        # ── 主干网络 ────────────────────────────────────────────────────────
        self.conv0_fluid = Conv('conv0_fluid', filters=layer_channels[0])
        self.conv0_obstacle = Conv('conv0_obstacle', filters=layer_channels[0])
        self.dense0_fluid = tf.keras.layers.Dense(units=layer_channels[0], name='dense0_fluid')

        self.convs: List[ml3d.layers.ContinuousConv] = []
        self.denses: List[tf.keras.layers.Dense] = []
        for i, ch in enumerate(layer_channels[1:], 1):
            self.denses.append(tf.keras.layers.Dense(units=ch, name=f'dense{i}'))
            self.convs.append(Conv(f'conv{i}', filters=ch))

        # ── 位置修正预测头 ──────────────────────────────────────────────────
        self.pos_conv = Conv('pos_conv', filters=3)
        self.pos_dense = tf.keras.layers.Dense(units=3, name='pos_dense')

        # ── VF 空间卷积（捕捉相分数在邻域间的传播）─────────────────────────
        # 输入: density-free VF branch features [N, C]  输出: [N, C]
        # 卷积聚合邻域 VF 状态，不再复用位置分支的 density-conditioned latent
        self.vf_context_conv = Conv('vf_context_conv', filters=layer_channels[-1])

        # ── VF 逐相预测器（按相共享 CConv 头）──────────────────────────────
        # 做法：把 [粒子, 相] 展平为 phase-wise 点云，并给不同相加固定位置偏移，
        # 这样同一套 CConv 权重只会在“同相邻域”里聚合，等价于对每一相共享空间预测器。
        # Dense 跳连保留局部自特征；CConv 提供真正的逐相空间传播。
        self.vf_per_phase_conv = Conv(
            'vf_per_phase_conv',
            filters=1,
            radius_search_ignore_query_points=False,
        )
        self.vf_per_phase_dense = tf.keras.layers.Dense(units=1, name='vf_per_phase_dense')

    # ─────────────────────────────────────────────────────────────────────────
    #  call
    # ─────────────────────────────────────────────────────────────────────────

    def call(self,
             inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
             current_num_phases: Optional[tf.Tensor] = None,
             phase_densities: Optional[tf.Tensor] = None,
             training: bool = False,
             **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        前向传播。

        Args:
            inputs: (pos1, vel1, current_phase_fractions, box_pos, box_feats)
                pos1                   : [N, 3]
                vel1                   : [N, 3]
                current_phase_fractions: [N, num_phases]  num_phases 可变
                box_pos                : [M, 3]
                box_feats              : [M, F]
            current_num_phases: 兼容旧接口保留，实际相数从输入张量形状读取
            phase_densities   : [num_phases] 各相密度；None 时全取 1000 kg/m³
            training          : 训练模式标志
            **kwargs          : cd (float), cf (float)

        Returns:
            (pos_final [N,3], vel_final [N,3], next_phase_fractions [N, num_phases])
        """
        pos1, vel1, current_phase_fractions, box_pos, box_feats = inputs

        # 实际相数直接从张量形状读取，不依赖 current_num_phases
        num_phases = tf.shape(current_phase_fractions)[1]

        # if phase_densities is None:
        #     phase_densities = tf.ones([num_phases], dtype=tf.float32) * 1000.0
        phase_densities = tf.convert_to_tensor(phase_densities, dtype=tf.float32)

        # 1. 物理积分
        pos2, vel2 = self.integrate_pos_vel(pos1, vel1)

        # 2. 相特征归一化（位置分支 DeepSet 与 VF 分支共享原始逐相输入）
        per_phase_features, phase_embedding = self._encode_phases(
            current_phase_fractions, phase_densities, num_phases
        )
        vf_only_features, vf_phase_embedding = self._encode_vf_branch(
            per_phase_features, num_phases
        )

        # 3. cd/cf 编码一次，仅供 VF 最终逐相决策层使用
        #    不再进入位置分支或 VF 空间上下文，避免条件污染局部动力学/邻域表征
        cd_cf_emb = None
        if self.cd_cf_as_input:
            # 显式校验，不传直接抛异常
            if 'cd' not in kwargs or 'cf' not in kwargs:
                raise ValueError("cd, cf must be provided in kwargs when cd_cf_as_input is True")
            # 严格必传：不传直接 KeyError 报错，程序停止
            cd = tf.cast(kwargs['cd'], dtype=tf.float32)
            cf = tf.cast(kwargs['cf'], dtype=tf.float32)
            cond = tf.reshape(tf.stack([cd, cf]), (1, 2))
            cd_cf_raw = self.cd_cf_encoder(cond)                        # [1, D]
            cd_cf_emb = tf.tile(cd_cf_raw, [tf.shape(pos2)[0], 1])     # [N, D]

        # 4. 构建粒子特征（主干网络用）
        # 位置分支不再直接接收 cd/cf，避免先由位置预测学会区分混溶/不混溶
        fluid_feats = self._build_fluid_feats(pos2, vel2, phase_embedding)

        # 5. 主干网络
        shared_features = self._backbone(fluid_feats, pos2, box_pos, box_feats)

        # 6. 位置修正
        filter_extent = tf.constant(self.filter_extent)
        pos_correction = (1.0 / 128.0) * (
            self.pos_conv(shared_features, pos2, pos2, filter_extent,
                          user_neighbors_index=self._fluid_nns_index,
                          user_neighbors_row_splits=self._fluid_nns_row_splits,
                          user_neighbors_importance=self._fluid_nns_importance)
            + self.pos_dense(shared_features)
        )
        pos_final, vel_final = self.compute_new_pos_vel(pos1, vel1, pos2, vel2, pos_correction)

        # 7. VF 分支独立上下文：只看可选 velocity + VF-only DeepSet
        vf_branch_feats = self._build_vf_feats(vel_final, vf_phase_embedding)

        # 8. VF 逐相预测（与位置分支解耦，避免密度经 shared_features 污染 VF）
        next_vf = self._predict_next_vf(
            vf_branch_feats, vf_only_features, current_phase_fractions, num_phases, pos_final,
            cd_cf_emb=cd_cf_emb
        )

        return pos_final, vel_final, next_vf

    # ─────────────────────────────────────────────────────────────────────────
    #  相特征编码
    # ─────────────────────────────────────────────────────────────────────────

    def _encode_phases(self,
                       phase_fractions: tf.Tensor,
                       phase_densities: tf.Tensor,
                       num_phases: tf.Tensor):
        """
        归一化相特征，返回：
          per_phase_features: [N, num_phases, 2]  逐相特征（供逐相预测器使用）
          phase_embedding   : [N, 64]             DeepSet 聚合嵌入（供主干网络使用）
        """
        densities_per_particle = tf.broadcast_to(
            phase_densities[:num_phases], tf.shape(phase_fractions)
        )
        log_densities = tf.math.log(densities_per_particle + 1e-8)

        if self.phase_feat_centralization:
            vf_scaled = (phase_fractions - 0.5) * 2.0
            log_density_scaled = (log_densities - 7.7) / 1.5
        else:
            vf_scaled = phase_fractions
            log_density_scaled = (log_densities - 6.2146) / 2.9957

        per_phase_features = tf.stack([vf_scaled, log_density_scaled], axis=-1)  # [N, P, 2]

        N = tf.shape(phase_fractions)[0]
        mask = tf.ones([N, num_phases], dtype=tf.bool)
        phase_embedding = self.phase_encoder((per_phase_features, mask))  # [N, 64]

        return per_phase_features, phase_embedding

    def _encode_vf_branch(self,
                          per_phase_features: tf.Tensor,
                          num_phases: tf.Tensor):
        """
        为 VF 分支构建仅含 VF 的置换不变编码。

        返回：
          vf_only_features : [N, num_phases, 1]  逐相 vf_scaled
          vf_phase_embedding: [N, 64]            DeepSet(VF-only) 聚合嵌入
        """
        vf_only_features = per_phase_features[..., 0:1]

        N = tf.shape(per_phase_features)[0]
        mask = tf.ones([N, num_phases], dtype=tf.bool)
        vf_phase_embedding = self.vf_phase_encoder((vf_only_features, mask))  # [N, 64]

        return vf_only_features, vf_phase_embedding

    # ─────────────────────────────────────────────────────────────────────────
    #  粒子特征构建
    # ─────────────────────────────────────────────────────────────────────────

    def _build_fluid_feats(self,
                           pos: tf.Tensor,
                           vel: tf.Tensor,
                           phase_embedding: tf.Tensor) -> tf.Tensor:
        """[N, 1 + 3 + 64 + cd_cf_dim]"""
        feats = [
            tf.ones_like(pos[:, 0:1]),  # [N, 1]
            vel,                         # [N, 3]
            phase_embedding,             # [N, 64]
        ]
        return tf.concat(feats, axis=-1)

    def _build_vf_feats(self,
                        vel: tf.Tensor,
                        vf_phase_embedding: tf.Tensor) -> tf.Tensor:
        """[N, (3 if vf_use_velocity else 0) + 64]，仅供 VF 分支空间上下文使用。"""
        feats = [
            vf_phase_embedding,         # [N, 64]
        ]
        if self.vf_use_velocity:
            feats.insert(0, vel)        # [N, 3]
        return tf.concat(feats, axis=-1)

    def _build_phasewise_positions(self,
                                   pos: tf.Tensor,
                                   num_phases: tf.Tensor) -> tf.Tensor:
        """
        将 [N, 3] 粒子坐标扩展为 [N * num_phases, 3] 的 phase-wise 坐标。

        通过给不同相添加固定偏移，确保逐相 CConv 只在“同相粒子”之间做邻域聚合，
        同时保持每一相内部的相对几何关系不变。
        """
        num_particles = tf.shape(pos)[0]
        repeated_pos = tf.tile(tf.expand_dims(pos, axis=1), tf.stack([1, num_phases, 1]))
        repeated_pos = tf.reshape(repeated_pos, [-1, 3])

        phase_ids = tf.tile(tf.expand_dims(tf.range(num_phases), axis=0), tf.stack([num_particles, 1]))
        phase_ids = tf.reshape(phase_ids, [-1])
        phase_spacing = tf.cast(self.filter_extent * 4.0, dtype=pos.dtype)
        phase_offsets = tf.cast(phase_ids, dtype=pos.dtype) * phase_spacing
        phase_offsets = tf.stack([
            phase_offsets,
            tf.zeros_like(phase_offsets),
            tf.zeros_like(phase_offsets),
        ], axis=-1)

        return repeated_pos + phase_offsets

    def _normalize_phase_fractions(self,
                                   vf_raw: tf.Tensor,
                                   fallback_vf: tf.Tensor) -> tf.Tensor:
        """
        使用 ReLU + 逐粒子归一化约束 VF；若某粒子的整行被裁成 0，则回退到当前 VF。
        """
        vf_non_negative = tf.nn.relu(vf_raw)
        vf_sum = tf.reduce_sum(vf_non_negative, axis=-1, keepdims=True)
        vf_normalized = vf_non_negative / tf.maximum(vf_sum, 1e-8)

        fallback_sum = tf.reduce_sum(fallback_vf, axis=-1, keepdims=True)
        fallback_normalized = fallback_vf / tf.maximum(fallback_sum, 1e-8)

        return tf.where(vf_sum > 1e-8, vf_normalized, fallback_normalized)

    # ─────────────────────────────────────────────────────────────────────────
    #  主干网络
    # ─────────────────────────────────────────────────────────────────────────

    def _backbone(self,
                  fluid_feats: tf.Tensor,
                  pos: tf.Tensor,
                  box_pos: tf.Tensor,
                  box_feats: tf.Tensor) -> tf.Tensor:
        filter_extent = tf.constant(self.filter_extent)

        x = tf.concat([
            self.conv0_obstacle(box_feats, box_pos, pos, filter_extent),
            self.conv0_fluid(fluid_feats, pos, pos, filter_extent),
            self.dense0_fluid(fluid_feats),
        ], axis=-1)

        # 缓存 pos→pos 邻居搜索结果，后续所有 fluid-fluid 卷积（同一 filter_extent）复用，
        # 避免重复 radius search（backbone 循环卷积 + pos_conv 共享此缓存）
        self._fluid_nns_index = self.conv0_fluid.nns.neighbors_index
        self._fluid_nns_row_splits = self.conv0_fluid.nns.neighbors_row_splits
        self._fluid_nns_importance = self.conv0_fluid._conv_values['neighbors_importance']

        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            tf.ones_like(self.conv0_fluid.nns.neighbors_index, dtype=tf.float32),
            self.conv0_fluid.nns.neighbors_row_splits,
        )

        ans = [x]
        for conv, dense in zip(self.convs, self.denses):
            inp = tf.keras.activations.relu(ans[-1])
            out = conv(inp, pos, pos, filter_extent,
                       user_neighbors_index=self._fluid_nns_index,
                       user_neighbors_row_splits=self._fluid_nns_row_splits,
                       user_neighbors_importance=self._fluid_nns_importance) + dense(inp)
            if out.shape[-1] == ans[-1].shape[-1]:
                out = out + ans[-1]
            ans.append(out)

        return tf.keras.activations.relu(ans[-1])

    # ─────────────────────────────────────────────────────────────────────────
    #  VF 逐相预测
    # ─────────────────────────────────────────────────────────────────────────

    def _predict_next_vf(self,
                         vf_branch_feats: tf.Tensor,
                         vf_only_features: tf.Tensor,
                         current_vf: tf.Tensor,
                         num_phases: tf.Tensor,
                         pos: tf.Tensor,
                         cd_cf_emb: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        逐相预测 VF 更新，权重在所有相之间共享。

        两种模式（由 self.vf_residual 控制）：
          直接预测（vf_residual=False，默认）：
            vf_next = normalize(relu(CConv_output))

          残差预测（vf_residual=True）：
            vf_next = normalize(relu(vf0 + alpha * delta_vf))

        两种模式均满足 vf ≥ 0 且 Σvf = 1（ReLU + 归一化保证）。
        VF 分支使用独立的 density-free 空间上下文，密度只通过位置更新、速度变化与邻域重排间接影响 VF。
        cd/cf 仅在最终逐相决策层注入，避免把全局条件提前混入空间上下文卷积。
        """
        filter_extent = tf.constant(self.filter_extent)

        # 空间卷积：让每个粒子在新位置处感知邻域 VF 状态
        vf_spatial = self.vf_context_conv(vf_branch_feats, pos, pos, filter_extent)  # [N, C]
        vf_spatial_expanded = tf.tile(
            tf.expand_dims(vf_spatial, axis=1), [1, num_phases, 1]
        )  # [N, num_phases, C]

        # 设计意图：
        #   vf_spatial_expanded 提供当前粒子在新位置处的邻域 VF 上下文（对该粒子的所有相共享）
        #   vf_only_features   提供当前被更新相的逐相身份/占比信号
        parts = [vf_spatial_expanded, vf_only_features]  # [N, P, C+1]
        if cd_cf_emb is not None:
            cd_cf_expanded = tf.tile(
                tf.expand_dims(cd_cf_emb, axis=1), [1, num_phases, 1]
            )  # [N, P, D]
            parts.append(cd_cf_expanded)
        per_phase_input = tf.concat(parts, axis=-1)  # [N, P, C+1(+D)]

        flat_per_phase_input = tf.reshape(per_phase_input, [-1, per_phase_input.shape[-1]])
        phasewise_pos = self._build_phasewise_positions(pos, num_phases)

        phasewise_prediction = self.vf_per_phase_conv(
            flat_per_phase_input, phasewise_pos, phasewise_pos, filter_extent
        ) + self.vf_per_phase_dense(flat_per_phase_input)
        phasewise_prediction = tf.reshape(
            tf.squeeze(phasewise_prediction, axis=-1),
            tf.shape(current_vf),
        )

        if self.vf_residual:
            vf_raw = current_vf + self.alpha * phasewise_prediction
        else:
            vf_raw = phasewise_prediction

        return self._normalize_phase_fractions(vf_raw, current_vf)

    # ─────────────────────────────────────────────────────────────────────────
    #  辅助方法
    # ─────────────────────────────────────────────────────────────────────────

    def integrate_pos_vel(self,
                          pos1: tf.Tensor,
                          vel1: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        dt = self.timestep
        vel2 = vel1 + dt * self.gravity
        pos2 = pos1 + dt * (vel1 + vel2) / 2.0
        return pos2, vel2

    def compute_new_pos_vel(self,
                            pos1: tf.Tensor,
                            vel1: tf.Tensor,
                            pos2_integrated: tf.Tensor,
                            vel2_integrated: tf.Tensor,
                            pos_correction: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        dt = self.timestep
        pos_final = pos2_integrated + pos_correction
        vel_final = (pos_final - pos1) / dt
        return pos_final, vel_final

    # ─────────────────────────────────────────────────────────────────────────
    #  初始化
    # ─────────────────────────────────────────────────────────────────────────

    def init(self, **kwargs) -> None:
        """用 2 相虚拟数据触发前向传播，完成所有权重初始化。"""
        pos = np.zeros((1, 3), dtype=np.float32)
        vel = np.zeros((1, 3), dtype=np.float32)
        phase_fractions = np.array([[1.0, 0.0]], dtype=np.float32)
        box = np.zeros((1, 3), dtype=np.float32)
        box_feats = np.zeros((1, 3), dtype=np.float32)
        densities = np.array([1000.0, 800.0], dtype=np.float32)

        _ = self.__call__(
            (pos, vel, phase_fractions, box, box_feats),
            phase_densities=tf.constant(densities, dtype=tf.float32),
            cd=np.float32(0.5),
            cf=np.float32(0.5),
        )
