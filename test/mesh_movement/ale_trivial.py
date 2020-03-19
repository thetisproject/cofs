from thetis import *

def solve_periodic(mode):
    """
    Solve a simple tracer advection problem with constant uniform fluid velocity parallel to the
    mesh periodicity.

    In Lagrangian mode, the mesh should be identical to how it was initially after one cycle.
    """
    assert mode in ('lagrangian', 'eulerian')

    # Set up domain
    n, L, T = 40, 10.0, 10.0
    mesh2d = PeriodicRectangleMesh(n, n, L, L)
    x, y = SpatialCoordinate(mesh2d)

    # Create solver object
    solver_obj = solver2d.FlowSolver2d(mesh2d, Constant(1.0))
    options = solver_obj.options
    options.timestepper_type = 'CrankNicolson'
    options.timestep = 0.2
    options.simulation_export_time = T/10
    options.simulation_end_time = T
    options.fields_to_export_hdf5 = []
    options.fields_to_export = ['tracer_2d']
    options.solve_tracer = True
    options.tracer_only = True
    options.horizontal_diffusivity = Constant(0.0)
    options.use_lagrangian_formulation = mode == 'lagrangian'

    # Constant uniform fluid velocity parallel to mesh periodicity
    fluid_velocity = Constant(as_vector([L/T, 0.0]))

    # Apply initial conditions
    solver_obj.create_function_spaces()
    init = project(exp(-((x-L/2)**2 + (y-L/2)**2)), solver_obj.function_spaces.P1DG_2d)
    init_coords = mesh2d.coordinates.copy(deepcopy=True)
    solver_obj.assign_initial_conditions(uv=fluid_velocity, tracer=init)

    # Apply boundary conditions
    neumann = {'diff_flux': Constant(0.0)}
    solver_obj.bnd_functions['tracer'] = {1: neumann, 2: neumann}

    # Solve and check advection cycle is complete
    solver_obj.iterate()
    final = solver_obj.fields.tracer_2d
    if mode == 'lagrangian':
        final_coords = mesh2d.coordinates
        final_coords -= Constant(as_vector([L, 0.0]))  # TODO: Account for periodicity
        final_coords -= init_coords
        try:
            tol = 1.0e-8
            assert final_coords.vector().gather().max() < tol
            assert final_coords.vector().gather().min() > -tol
        except AssertionError:
            raise ValueError("Initial and final mesh coordinates do not match.")
    msg = "Relative error in '{:s}' mode: {:.4f}%"
    print_output(msg.format(mode, 100*abs(errornorm(init, final)/norm(init))))


if __name__ == "__main__":
    solve_periodic('lagrangian')
    # solve_periodic('eulerian')
