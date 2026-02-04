def set_science_style():
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    plt.style.use(["science"])
    plt.rcParams["text.usetex"] = False
    plt.rcParams["svg.fonttype"] = "none"
