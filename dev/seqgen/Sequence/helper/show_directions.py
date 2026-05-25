#def calculate_bfactor():
    #b_factor(i)=(norm(directions(i,:)))^2


def show_on_sphere(dirs):
    """
    Show direction vectors on a sphere
    """

    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    class Arrow3D(FancyArrowPatch):

        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.set_aspect("equal")

    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")

    # draw a point
    ax.scatter([0], [0], [0], color="g", s=100)

    # draw a vector
    def make_arrow(x, y, z):
        return Arrow3D([0, x], [0, y], [0, z], mutation_scale=20,
                    lw=2, arrowstyle="-|>", color="k")

    for x, y, z in dirs(6):
        ax.add_artist(make_arrow(+x, +y, +z))
        ax.add_artist(make_arrow(-x, -y, -z))


if __name__ == "__main__":
    dirs = get_directions(6)
    print(dirs)

    show_on_sphere(dirs)
    plt.show()
