import tensorflow as tf
import open3d.ml.tf as ml3d
import numpy as np
from debug_utils import debug_print


class MultiPhaseParticleNetwork(tf.keras.Model):

    def __init__(self,
                 kernel_size=[4, 4, 4],
                 radius_scale=1.5,
                 coordinate_mapping='ball_to_cube_volume_preserving',
                 interpolation='linear',
                 use_window=True,
                 particle_radius=0.05,
                 timestep=1 / 50,
                 gravity=(0, -9.81, 0),
                 num_phases=2,
                 cd_cf_as_input=True,
                 cd_cf_embedding_dim=16):
        super().__init__(name=type(self).__name__)

        self.num_phases = num_phases
        # 网络输出: 3 (位置修正) + num_phases (每个相的logits)
        self.layer_channels = [32, 64, 64, 3 + self.num_phases]

        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.filter_extent = np.float32(self.radius_scale * 6 * self.particle_radius)
        self.timestep = timestep
        self.gravity = tf.constant(gravity, dtype=tf.float32) # Make gravity a constant tensor

        self.cd_cf_as_input = cd_cf_as_input
        self.cd_cf_embedding_dim = cd_cf_embedding_dim

        debug_print(f"Particle Radius: {self.particle_radius}")
        debug_print(f"Filter Extent: {self.filter_extent}")
        debug_print(f"Number of Phases: {self.num_phases}")
        debug_print(f"Output Layer Channels: {self.layer_channels[-1]}")


        if self.cd_cf_as_input and self.cd_cf_embedding_dim > 0:
            self.cd_embedding_layer = tf.keras.layers.Dense(self.cd_cf_embedding_dim,
                                                            activation='tanh', # Or 'relu' or None
                                                            name='cd_embedding')
            self.cf_embedding_layer = tf.keras.layers.Dense(self.cd_cf_embedding_dim,
                                                            activation='tanh', # Or 'relu' or None
                                                            name='cf_embedding')

        self._all_convs = []

        def window_poly6(r_sqr):
            # Ensure r_sqr is not negative, which can happen due to precision issues
            r_sqr_clipped = tf.maximum(r_sqr, 0.0)
            return tf.clip_by_value((1 - r_sqr_clipped)**3, 0, 1)


        def Conv(name, activation=None, **kwargs):
            conv_fn = ml3d.layers.ContinuousConv
            window_fn = None
            if self.use_window:
                window_fn = window_poly6

            conv = conv_fn(name=name,
                           kernel_size=self.kernel_size,
                           activation=activation,
                           align_corners=True,
                           interpolation=self.interpolation,
                           coordinate_mapping=self.coordinate_mapping,
                           normalize=False,
                           window_function=window_fn,
                           radius_search_ignore_query_points=True,
                           **kwargs)
            self._all_convs.append((name, conv))
            return conv

        self.conv0_fluid = Conv(name="conv0_fluid",
                                filters=self.layer_channels[0],
                                activation=None)
        self.conv0_obstacle = Conv(name="conv0_obstacle",
                                   filters=self.layer_channels[0],
                                   activation=None)
        self.dense0_fluid = tf.keras.layers.Dense(name="dense0_fluid",
                                                  units=self.layer_channels[0],
                                                  activation=None)

        self.convs = []
        self.denses = []
        for i in range(1, len(self.layer_channels)):
            ch = self.layer_channels[i]
            # Use a different name for the last dense layer if its purpose is different
            dense_name = f"dense{i}" if i < len(self.layer_channels) -1 else "dense_output"
            conv_name = f"conv{i}" if i < len(self.layer_channels) -1 else "conv_output"

            dense = tf.keras.layers.Dense(units=ch, name=dense_name, activation=None)
            # The last conv layer might not need an activation if followed by softmax or linear output
            conv_activation = None # tf.keras.activations.relu if i < len(self.layer_channels) - 1 else None
            conv = Conv(name=conv_name, filters=ch, activation=conv_activation)
            self.denses.append(dense)
            self.convs.append(conv)

    def integrate_pos_vel(self, pos1, vel1):
        dt = self.timestep
        vel2 = vel1 + dt * self.gravity
        pos2 = pos1 + dt * (vel1 + vel2) / 2.0 # More stable: use average velocity over timestep
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2_integrated, vel2_integrated, pos_correction):
        dt = self.timestep
        # Apply correction to the integrated position
        pos_final = pos2_integrated + pos_correction
        # Velocity is based on the change from original position to final corrected position
        vel_final = (pos_final - pos1) / dt
        return pos_final, vel_final

    def compute_next_phase_fractions(self, current_phase_fractions, network_vf_logits):
        """
        Computes the next phase fractions using softmax for stability.
        Args:
            current_phase_fractions: Tensor [batch_size, num_particles, num_phases], current VFs.
                                     Used for potential residual connection if desired.
            network_vf_logits: Tensor [batch_size, num_particles, num_phases], raw logits from the network.
        """
        if self.num_phases <= 1:
            return current_phase_fractions

        # Option 1: Network predicts new logits directly
        # For stability, it can be beneficial if the network predicts a *change* to the logits
        # or if the logits are somehow scaled relative to the current state.
        # For now, let's assume network_vf_logits are the new logits.
        
        # Residual connection to logits (optional, can help learning identity for stable regions)
        # One way to add residual: transform current_phase_fractions to a logit-like space
        # inverse_softmax_approx = tf.math.log(current_phase_fractions + 1e-8) # Add epsilon
        # combined_logits = inverse_softmax_approx + network_vf_logits # Network learns a delta in logit space
        
        # Or simpler: network directly predicts the logits for the next step
        combined_logits = network_vf_logits

        # Apply softmax to get normalized, non-negative phase fractions
        next_fractions = tf.nn.softmax(combined_logits, axis=-1)
        
        return next_fractions

    def compute_correction_and_vf_logits(self,
                                         pos, # Current (integrated) particle positions
                                         vel, # Current (integrated) particle velocities
                                         phase_fractions, # All N phases [batch, num_particles, num_phases]
                                         box_pos,
                                         box_feats,
                                         fixed_radius_search_hash_table=None,
                                         cd_scalar=0.5,
                                         cf_scalar=0.5):
        filter_extent_tensor = tf.constant(self.filter_extent, dtype=tf.float32)

        # --- Feature Engineering ---
        fluid_feats_list = [
            tf.ones_like(pos[:, 0:1]), # Existence feature
            vel # Current velocity
        ]

        if self.num_phases > 1 and phase_fractions is not None:
            fluid_feats_list.append(phase_fractions) # Raw phase fractions

            # Placeholder for phase gradient features (calculated externally or via SPH ops)
            # These are CRUCIAL for the network to "see" interfaces and pure regions.
            # Example:
            # phase_gradients = compute_sph_gradient_for_all_phases(phase_fractions, pos, neighbors_info)
            # fluid_feats_list.append(phase_gradients) # Shape: [batch, num_particles, num_phases * 3]

            # Example: Non-linear transformations (use with caution, can cause instability if not scaled)
            # fluid_feats_list.append(phase_fractions**2)
            # fluid_feats_list.append(tf.math.sqrt(phase_fractions + 1e-8))


        if self.cd_cf_as_input:
            cd_tensor_val = tf.cast(cd_scalar, dtype=tf.float32)
            cf_tensor_val = tf.cast(cf_scalar, dtype=tf.float32)
            batch_size = tf.shape(pos)[0]

            if self.cd_cf_embedding_dim > 0:
                # Create per-particle tensors for embedding
                cd_per_particle = tf.ones((batch_size, 1), dtype=tf.float32) * cd_tensor_val
                cf_per_particle = tf.ones((batch_size, 1), dtype=tf.float32) * cf_tensor_val
                
                cd_embed = self.cd_embedding_layer(cd_per_particle)
                cf_embed = self.cf_embedding_layer(cf_per_particle)
                fluid_feats_list.extend([cd_embed, cf_embed])
            else:
                cd_direct = tf.ones_like(pos[:, 0:1]) * cd_tensor_val
                cf_direct = tf.ones_like(pos[:, 0:1]) * cf_tensor_val
                fluid_feats_list.extend([cd_direct, cf_direct])

        fluid_feats = tf.concat(fluid_feats_list, axis=-1)
        debug_print("Shape of fluid_feats fed to conv0_fluid: ", tf.shape(fluid_feats))


        # --- Network Forward Pass ---
        self.ans_conv0_fluid = self.conv0_fluid(fluid_feats, pos, pos, filter_extent_tensor)
        self.ans_dense0_fluid = self.dense0_fluid(fluid_feats) # Acts on original fluid_feats
        
        # Obstacle interaction (ensure box_pos and box_feats are correctly shaped)
        self.ans_conv0_obstacle = self.conv0_obstacle(box_feats, box_pos, pos, filter_extent_tensor)
        processed_feats = tf.concat([self.ans_conv0_obstacle, self.ans_conv0_fluid, self.ans_dense0_fluid], axis=-1)

        self.ans_convs = [processed_feats]
        for i, (conv_layer, dense_layer) in enumerate(zip(self.convs, self.denses)):
            # Apply ReLU to all intermediate features
            # The very last layer's output (logits) should typically not have ReLU if followed by softmax
            current_features = self.ans_convs[-1]
            current_features = tf.keras.activations.relu(current_features)

            ans_conv = conv_layer(current_features, pos, pos, filter_extent_tensor)
            ans_dense = dense_layer(current_features) # Acts on the (potentially ReLU'd) features

            # Skip connection logic
            if ans_dense.shape[-1] == self.ans_convs[-1].shape[-1]:
                # Residual connection for layers with same channel size
                # Ensure current_features (from self.ans_convs[-1]) is added before potential ReLU
                # if the skip is meant to be from the pre-activation state.
                # Here, it's simpler: adds to the post-processed (potentially ReLU'd) features.
                ans = ans_conv + ans_dense + self.ans_convs[-1] # Add previous layer's output
            else:
                ans = ans_conv + ans_dense
            self.ans_convs.append(ans)

        # --- Output Processing ---
        raw_network_output = self.ans_convs[-1] # Output from the last (conv+dense) block

        pos_correction = (1.0 / 128.0) * raw_network_output[..., :3] # Scale position correction

        vf_logits = None
        if self.num_phases > 1:
            # vf_logits are the raw outputs for phase fractions, to be softmaxed later
            # No large scaling factor here initially, let softmax handle normalization.
            # Scaling might be learned by the network or applied if logits become too large/small.
            vf_logits = raw_network_output[..., 3:]
            debug_print("Shape of vf_logits from network: ", tf.shape(vf_logits))

        # For debugging/loss calculation: neighbor counts
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            tf.ones_like(self.conv0_fluid.nns.neighbors_index, dtype=tf.float32),
            self.conv0_fluid.nns.neighbors_row_splits)
        # Similar for box_neighbors if needed for loss

        return pos_correction, vf_logits


    def call(self, inputs, training=False, fixed_radius_search_hash_table=None, cd=0.5, cf=0.5):
        pos1, vel1, current_phase_fractions, box_pos, box_feats = inputs
        # Ensure current_phase_fractions has shape [batch, num_particles, num_phases]

        # 1. Integrate position and velocity (e.g., with gravity)
        pos2_integrated, vel2_integrated = self.integrate_pos_vel(pos1, vel1)

        # 2. Compute network corrections and new VF logits
        # Pass integrated pos/vel and current VFs to the network
        pos_correction, next_vf_logits = self.compute_correction_and_vf_logits(
            pos2_integrated,
            vel2_integrated,
            current_phase_fractions, # Current VFs used as input features
            box_pos,
            box_feats,
            fixed_radius_search_hash_table,
            cd_scalar=cd,
            cf_scalar=cf)

        # 3. Apply position correction and compute final velocity
        pos_final, vel_final = self.compute_new_pos_vel(
            pos1, vel1, pos2_integrated, vel2_integrated, pos_correction)

        # 4. Compute next phase fractions using logits
        next_phase_fractions_final = current_phase_fractions # Default for single phase
        if self.num_phases > 1 and next_vf_logits is not None:
            next_phase_fractions_final = self.compute_next_phase_fractions(
                current_phase_fractions, # Can be used for residual logits
                next_vf_logits)
        
        return pos_final, vel_final, next_phase_fractions_final

    def init(self, feats_shape=None):
        """Initializes the model by doing a forward pass with dummy data."""
        pos = np.zeros(shape=(1, 3), dtype=np.float32)
        vel = np.zeros(shape=(1, 3), dtype=np.float32)

        if self.num_phases > 1:
            phase_fractions = np.zeros(
                shape=(1, self.num_phases), dtype=np.float32)
            # 初始设置第一相的分数为1
            phase_fractions[:, 0] = 1.0
        else:
            phase_fractions = None # Or a dummy tensor if your 'call' expects something

        box = np.zeros(shape=(1, 3), dtype=np.float32)
        box_feats = np.zeros(shape=(1, 3), dtype=np.float32)

        cd = np.float32(0.5)
        cf = np.float32(0.5)
        # Call the model to build layers
        _ = self.__call__((pos, vel, phase_fractions,
                          box, box_feats), cd=cd, cf=cf)
        print(f"{self.name} initialized with {self.num_phases} phases.")
        print("Model summary (if not too complex for console):")
        try:
            self.summary() # Might be very long for graph networks
        except Exception as e:
            print(f"Could not print model summary: {e}")