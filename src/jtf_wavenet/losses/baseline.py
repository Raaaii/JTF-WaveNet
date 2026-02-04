import tensorflow as tf


class BaselineSTD(tf.Module):
    def __init__(self, number_of_bins=100, localised=False, apply_softplus_inverse=False):
        self.number_of_bins = int(number_of_bins)
        self.localised = localised
        self.apply_softplus_inverse = apply_softplus_inverse

    @staticmethod
    def inverse_softplus(y, epsilon=1e-8):
        return tf.math.log(tf.math.expm1(y) + epsilon)

    def calc_bin_mean(self, data):
        print("\n[calc_bin_mean] input data shape:", data.shape)
        N = tf.shape(data)[0] // self.number_of_bins
        binned_data = tf.reshape(data, (self.number_of_bins, N))
        # mean_per_bin = tf.reduce_mean(binned_data, axis=1)

        mean_per_bin = tf.reduce_mean(tf.abs(binned_data), axis=1)
        print("[calc_bin_mean] number_of_bins:", self.number_of_bins)
        print("[calc_bin_mean] N (points per bin):", N)
        print("[calc_bin_mean] binned_data shape:", binned_data.shape)
        print("[calc_bin_mean] mean_per_bin shape:", mean_per_bin.shape)
        return mean_per_bin, binned_data, N

    def select_lowest_half_bins(self, data):
        print("\n[select_lowest_half_bins] input data shape:", data.shape)
        mean_per_bin, binned_data, N = self.calc_bin_mean(data)

        half_bins = self.number_of_bins // 2  # 50% lowest
        sorted_indices = tf.argsort(mean_per_bin, direction="ASCENDING")
        low_bin_indices = sorted_indices[:half_bins]
        selected_bins = tf.gather(binned_data, low_bin_indices)

        print("[select_lowest_half_bins] mean_per_bin shape:", mean_per_bin.shape)
        print("[select_lowest_half_bins] sorted_indices shape:", sorted_indices.shape)
        print("[select_lowest_half_bins] low_bin_indices shape:", low_bin_indices.shape)
        print("[select_lowest_half_bins] selected_bins shape:", selected_bins.shape)

        bin_offsets = tf.expand_dims(low_bin_indices * N, axis=1)  # [half_bins,1]
        in_bin_offsets = tf.range(N)
        in_bin_offsets = tf.reshape(in_bin_offsets, (1, N))
        selected_indices = bin_offsets + in_bin_offsets
        selected_indices_flat = tf.reshape(selected_indices, [-1])
        selected_points_flat = tf.reshape(selected_bins, [-1])

        print("[select_lowest_half_bins] N:", N)
        print("[select_lowest_half_bins] bin_offsets shape:", bin_offsets.shape)
        print("[select_lowest_half_bins] in_bin_offsets shape:", in_bin_offsets.shape)
        print("[select_lowest_half_bins] selected_indices shape:", selected_indices.shape)
        print("[select_lowest_half_bins] selected_indices_flat shape:", selected_indices_flat.shape)
        print("[select_lowest_half_bins] selected_points_flat shape:", selected_points_flat.shape)

        return selected_points_flat, selected_indices_flat

    def write_std_map(self, data, selected_points, selected_indices):
        print("\n[write_std_map] data shape:", data.shape)
        print("[write_std_map] selected_points shape:", selected_points.shape)
        print("[write_std_map] selected_indices shape:", selected_indices.shape)

        std_val = tf.math.reduce_std(selected_points)
        if self.apply_softplus_inverse:
            std_val = self.inverse_softplus(std_val)

        print("[write_std_map] std_val:", std_val)

        if self.localised:
            std_map = tf.zeros_like(data, dtype=tf.float32)
            updates = tf.fill(tf.shape(selected_indices), std_val)
            std_map = tf.tensor_scatter_nd_update(
                std_map, tf.expand_dims(selected_indices, 1), updates
            )
        else:
            std_map = tf.ones_like(data, dtype=tf.float32) * std_val

        print("[write_std_map] std_map shape:", std_map.shape)
        return std_map

    def get_std_map(self, data):
        print("\n[get_std_map] input shape:", data.shape)
        selected_points, selected_indices = self.select_lowest_half_bins(data)
        std_map = self.write_std_map(data, selected_points, selected_indices)
        print("[get_std_map] output std_map shape:", std_map.shape)
        return std_map

    def batch_std_map(self, batched_data):
        print("\n[batch_std_map] raw batched_data shape:", batched_data.shape)

        # Adjust to expected layout [B, 4096]
        batched_data = batched_data[:, 0, :, 0]
        print("[batch_std_map] after slice ->", batched_data.shape)

        std_map = tf.map_fn(self.get_std_map, batched_data)
        print("[batch_std_map] after map_fn ->", std_map.shape)

        std_map = std_map[:, tf.newaxis, :, tf.newaxis]
        print("[batch_std_map] final std_map shape:", std_map.shape)

        return std_map
