# Idealised channel flow in 3D
# ============================
#
# Solves shallow water equations in closed rectangular domain
# with sloping bathymetry.
#
# Flow is forced with tidal volume flux in the deep (ocean) end of the
# channel, and a constant volume flux in the shallow (river) end.
#
# This test is useful for testing open boundary conditions.
#
# Tuomas Karna 2015-03-03
from cofs import *

import pytest


@pytest.mark.skipif(True, reason='test is obsolete')
def test_closed_channel():
    n_layers = 6
    outputDir = createDirectory('outputs')
    mesh2d = Mesh('mesh_coarse.msh')
    printInfo('Loaded mesh '+mesh2d.name)
    printInfo('Exporting to '+outputDir)
    T = 24 * 3600
    Umag = Constant(2.5)
    TExport = 300.0

    # bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')

    depth_oce = 20.0
    depth_riv = 7.0
    bathymetry_2d.interpolate(Expression('ho - (ho-hr)*x[0]/100e3',
                                         ho=depth_oce, hr=depth_riv))

    # create solver
    solverObj = solver.flowSolver(mesh2d, bathymetry_2d, n_layers)
    options = solverObj.options
    # options.nonlin = False
    options.solveSalt = False
    options.solveVertDiffusion = False
    options.useBottomFriction = False
    options.useALEMovingMesh = False
    options.uvLaxFriedrichs = Constant(1.0)
    options.tracerLaxFriedrichs = Constant(1.0)
    # options.useSemiImplicit2D = False
    # options.useModeSplit = False
    # options.baroclinic = True
    options.TExport = TExport
    options.T = T
    options.outputDir = outputDir
    options.uAdvection = Umag
    options.checkSaltDeviation = True
    options.timerLabels = ['mode2d', 'momentumEq', 'vert_diffusion']
    options.fieldsToExport = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                              'w_3d', 'w_mesh_3d', 'salt_3d',
                              'baroc_head_3d', 'baroc_head_2d',
                              'uv_dav_2d', 'uv_bottom_2d']

    # initial conditions
    salt_init3d = Constant(4.5)

    # weak boundary conditions
    L_y = 20e3
    un_amp = -2.0
    flux_amp = L_y*depth_oce*un_amp
    h_T = 12 * 3600  # 44714.0
    un_river = -0.3
    flux_river = L_y*depth_riv*un_river
    t = 0.0
    T_ramp = 100.0
    # python function that returns time dependent boundary values
    ocean_flux_func = lambda t: (flux_amp*sin(2 * pi * t / h_T) -
                                 flux_river)*min(t/T_ramp, 1.0)
    river_flux_func = lambda t: flux_river*min(t/T_ramp, 1.0)
    # Constants that will be fed to the model
    ocean_flux = Constant(ocean_flux_func(t))
    river_flux = Constant(river_flux_func(t))

    # boundary conditions are defined with a dict
    # key defines the type of bnd condition, value the necessary coefficient(s)
    # here setting outward bnd flux (positive outward)
    ocean_funcs = {'flux': ocean_flux}
    river_funcs = {'flux': river_flux}
    ocean_funcs_3d = {'flux': ocean_flux}
    river_funcs_3d = {'flux': river_flux}
    # and constant salinity (for inflow)
    ocean_salt_3d = {'value': salt_init3d}
    river_salt_3d = {'value': salt_init3d}
    # bnd conditions are assigned to each boundary tag with another dict
    ocean_tag = 2
    river_tag = 1
    # assigning conditions for each equation
    # these must be assigned before equations are created
    solverObj.bnd_functions['shallow_water'] = {ocean_tag: ocean_funcs,
                                                river_tag: river_funcs}
    solverObj.bnd_functions['momentum'] = {2: ocean_funcs_3d, 1: river_funcs_3d}
    solverObj.bnd_functions['salt'] = {2: ocean_salt_3d, 1: river_salt_3d}

    def updateForcings(t_new):
        """Callback function that updates all time dependent forcing fields
        for the 2d mode"""
        ocean_flux.assign(ocean_flux_func(t_new))
        river_flux.assign(river_flux_func(t_new))

    def updateForcings3d(t_new):
        """Callback function that updates all time dependent forcing fields
        for the 3D mode"""
        ocean_flux.assign(ocean_flux_func(t_new))
        river_flux.assign(river_flux_func(t_new))

    # set init conditions, this will create all function spaces, equations etc
    solverObj.assignInitialConditions(salt=salt_init3d)
    solverObj.iterate(updateForcings=updateForcings,
                      updateForcings3d=updateForcings3d)

if __name__ == '__main__':
    test_closed_channel()