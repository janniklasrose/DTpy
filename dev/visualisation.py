import pyvista as pv

from DTpy import DiffusionTensor

# https://docs.pyvista.org/examples/01-filter/glyphs.html#sphx-glr-examples-01-filter-glyphs-py
# https://docs.pyvista.org/examples/01-filter/glyphs_table.html#sphx-glr-examples-01-filter-glyphs-table-py

tensors = [
    DiffusionTensor([1, 0, 0, 0, 2, 0.5, 3]),
    DiffusionTensor([1, 0, 0, 0, 2, 1.2, 3]),
    DiffusionTensor([4, 0, 0, 0, 2, 0.5, 3]),
    DiffusionTensor([3, 0, 0, 0, 2, 0.5, 3]),
]

# get dataset where to put glyphs
x, y, z = np.mgrid[:1, :1, :1]
mesh = pv.StructuredGrid(x, y, z)
mesh.point_data['scalars'] = [t.MD for t in tensors]

# construct the glyphs on top of the mesh
glyphs = mesh.glyph(
    geom=[pv.Superquadric(scale=t.EigenValues) for t in tensors],
    indices=np.arange(len(tensors)),  # values for scalars to look up glyphs by
    scale=False,  # don't scale by value of scalars
    #orient="vectors",  # ????
    #factor=0.3,  # ????
    #rng=(0, N-1),  # ????
)

# create plotter and add our glyphs with some nontrivial lighting
plotter = pv.Plotter()
plotter.add_mesh(glyphs, specular=1, specular_power=15,
                 smooth_shading=True, show_scalar_bar=False)
plotter.show()


def write_tensor_vtk(
    filename,
    points: list[tuple[float, float, float]],
    tensors: list[list[tuple[float, float, float]]],
    *,  # keyword-args only after here
    comment: str = "Example",
):
    num_pts = len(points)
    tensor_name = "DiffusionTensor"
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0")
        f.write(comment)
        f.write("ASCII")
        f.write("DATASET POLYDATA")
        f.write(f"POINTS {num_pts} float")
        for x, y, z in points:
            f.write(f"{x} {y} {z}")
        f.write(f"POINT_DATA {num_pts}")
        f.write(f"TENSORS {tensor_name} float")
        for tensor in tensors:
            for x, y, z in tensor:
                f.write("{x} {y} {z}")


if __name__ == "__main__":
    point = (0, 0, 0)
    tensor = [
        (1  , 0, 0.5),  # xx, xy, xz
        (0  , 1, 0  ),  # yx, yy, yz
        (0.5, 0, 2  ),  # zx, zy, zz
    ]
    #write_tensor_vtk("test.vtk", [point], [tensor], comment="Demo tensor")
