"""
Solve a simple tracer advection problem in a singly periodic domain with constant uniform fluid
velocity parallel to the mesh periodicity.

In Lagrangian mode, the mesh should be identical to how it was initially after one cycle.
"""
from thetis import *
import pytest
import os


def run(mode, **model_options):
    assert mode in ('lagrangian', 'eulerian')
    print_output("--- running {:s} mode".format(mode.capitalize()))

    # Set up domain
    n, L, T = 40, 10.0, 10.0
    mesh2d = PeriodicRectangleMesh(n, n, L, L)
    x, y = SpatialCoordinate(mesh2d)

    # Create solver object
    solver_obj = solver2d.FlowSolver2d(mesh2d, Constant(1.0))
    options = solver_obj.options
    options.timestep = 0.2
    options.simulation_export_time = T/10
    options.simulation_end_time = T
    options.fields_to_export_hdf5 = []
    options.no_exports = True
    options.solve_tracer = True
    options.tracer_only = True
    options.horizontal_diffusivity = None
    options.use_ale_moving_mesh_2d = mode == 'lagrangian'
    options.update(model_options)

    # Constant uniform fluid velocity parallel to mesh periodicity
    fluid_velocity = Constant(as_vector([L/T, 0.0]))

    # Apply initial conditions
    solver_obj.create_function_spaces()
    init = project(exp(-((x-L/2)**2 + (y-L/2)**2)), solver_obj.function_spaces.P1DG_2d)
    init_coords = mesh2d.coordinates.copy(deepcopy=True)
    solver_obj.assign_initial_conditions(uv=fluid_velocity, tracer=init)
    init = solver_obj.fields.tracer_2d.copy(deepcopy=True)

    # Apply boundary conditions
    neumann = {'diff_flux': Constant(0.0)}
    solver_obj.bnd_functions['tracer'] = {1: neumann, 2: neumann}

    def get_fluid_velocity(t):
        uv, eta = solver_obj.fields.solution_2d.split()
        return uv

    prescribed_velocity = None if mode == 'eulerian' else get_fluid_velocity

    # Solve and check advection cycle is complete
    solver_obj.iterate(prescribed_velocity=prescribed_velocity)
    final = solver_obj.fields.tracer_2d
    if mode == 'lagrangian':
        final_coords = mesh2d.coordinates
        final_coords -= Constant(as_vector([L, 0.0]))  # TODO: Account for periodicity
        coord_diff = final_coords.copy(deepcopy=True)
        coord_diff -= init_coords
        try:
            tol = 1.0e-8
            assert coord_diff.vector().gather().max() < tol
            assert coord_diff.vector().gather().min() > -tol
        except AssertionError:
            raise ValueError("Initial and final mesh coordinates do not match.")
    error = abs(errornorm(init, final)/norm(init))
    final -= init
    if not options.no_exports:
        File(os.path.join(options.output_directory, '{:s}_error.pvd'.format(mode))).write(final)
    print_output("Relative error in {:s} mode: {:.4f}%".format(mode.capitalize(), 100*error))
    return error  # FIXME: Surely error should be zero for Lagrangian mode?

# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.mark.parametrize(('stepper'),
                         [('CrankNicolson')])
def test_lagrangian_vs_eulerian(stepper):
    eulerian_error = run('eulerian', timestepper_type=stepper)
    lagrangian_error = run('lagrangian', timestepper_type=stepper)
    assert lagrangian_error < eulerian_error

# ---------------------------
# run individual setup for debugging
# ---------------------------

if __name__ == "__main__":
    run('eulerian', no_exports=False, fields_to_export=['tracer_2d'])
    run('lagrangian', no_exports=False, fields_to_export=['tracer_2d'])
