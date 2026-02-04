from jtf_wavenet.data.parameter_sampling import load_config
from jtf_wavenet.vis.generator_checks import plot_all_three_in_one_pass

if __name__ == "__main__":
    cfg = load_config("configs/default_generator.json")
    plot_all_three_in_one_pass(cfg)
