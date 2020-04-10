"""
Implements Equation and Term classes.

"""
from __future__ import absolute_import
from .utility import *
from collections import OrderedDict


class Term(object):
    """
    Implements a single term of an equation.

    .. note::
        Sign convention: all terms are assumed to be on the right hand side of the equation: d(u)/dt = term.
    """
    def __init__(self, function_space):
        """
        :arg function_space: the :class:`FunctionSpace` the solution belongs to
        """
        # define bunch of members needed to construct forms
        self.function_space = function_space
        self.mesh = self.function_space.mesh()
        self.test = TestFunction(self.function_space)
        self.tri = TrialFunction(self.function_space)
        self.normal = FacetNormal(self.mesh)
        # TODO construct them here from mesh ?
        self.boundary_markers = sorted(function_space.mesh().exterior_facets.unique_markers)
        self.boundary_len = function_space.mesh().boundary_len

    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions):
        """
        Returns an UFL form of the term.

        :arg solution: solution :class:`.Function` of the corresponding equation
        :arg solution_old: a time lagged solution :class:`.Function`
        :arg fields: a dictionary that provides all the remaining fields that the term depends on.
            The keys of the dictionary should standard field names in `field_metadata`
        :arg fields_old: Time lagged dictionary of fields
        :arg bnd_conditions: A dictionary describing boundary conditions.
            E.g. {3: {'elev_2d': Constant(1.0)}} replaces elev_2d function by a constant on boundary ID 3.
        """
        raise NotImplementedError('Must be implemented in the derived class')

    def jacobian(self, solution, solution_old, fields, fields_old, bnd_conditions):
        """
        Returns an UFL form of the Jacobian of the term.

        :arg solution: solution :class:`.Function` of the corresponding equation
        :arg solution_old: a time lagged solution :class:`.Function`
        :arg fields: a dictionary that provides all the remaining fields that the term depends on.
            The keys of the dictionary should standard field names in `field_metadata`
        :arg fields_old: Time lagged dictionary of fields
        :arg bnd_conditions: A dictionary describing boundary conditions.
            E.g. {3: {'elev_2d': Constant(1.0)}} replaces elev_2d function by a constant on boundary ID 3.
        """
        # TODO default behavior: symbolic expression, or implement only if user-defined?
        raise NotImplementedError('Must be implemented in the derived class')


class Equation(object):
    """
    Implements an equation, made out of terms.
    """
    SUPPORTED_LABELS = frozenset(['source', 'explicit', 'implicit', 'nonlinear'])
    """
    Valid labels for terms, indicating how they should be treated in the time
    integrator.

    source
        The term is a source term, i.e. does not depend on the solution.

    explicit
        The term should be treated explicitly

    implicit
        The term should be treated implicitly

    nonlinear
        The term is nonlinear and should be treated fully implicitly
    """

    def __init__(self, function_space):
        """
        :arg function_space: the :class:`FunctionSpace` the solution belongs to
        """
        self.terms = OrderedDict()
        self.labels = {}
        self.function_space = function_space
        self.mesh = self.function_space.mesh()
        self.test = TestFunction(self.function_space)
        self.trial = TrialFunction(self.function_space)
        # mesh dependent variables
        self.normal = FacetNormal(self.mesh)
        self.xyz = SpatialCoordinate(self.mesh)
        self.e_x, self.e_y, self.e_y = unit_vectors(3)

    def mass_term(self, solution):
        """
        Returns default mass matrix term for the solution function space.

        :returns: UFL form of the mass term
        """
        return inner(solution, self.test) * dx

    def add_term(self, term, label):
        """
        Adds a term in the equation

        :arg term: :class:`.Term` object to add_term
        :arg string label: Assign a label to the term. Valid labels are given by
            :attr:`.SUPPORTED_LABELS`.
        """
        key = term.__class__.__name__
        self.terms[key] = term
        self.label_term(key, label)

    def label_term(self, term, label):
        """
        Assings a label to the given term(s).

        :arg term: :class:`.Term` object, or a tuple of terms
        :arg label: string label to assign
        """
        if isinstance(term, str):
            assert term in self.terms, 'Unknown term, add it to the equation'
            assert label in self.SUPPORTED_LABELS, 'bad label: {:}'.format(label)
            self.labels[term] = label
        else:
            for k in iter(term):
                self.label_term(k, label)

    def select_terms(self, label):
        """
        Generator function that selects terms by label(s).

        label can be a single label (e.g. 'explicit'), 'all' or a tuple of
        labels.
        """
        if isinstance(label, str):
            if label == 'all':
                labels = self.SUPPORTED_LABELS
            else:
                labels = frozenset([label])
        else:
            labels = frozenset(label)
        for key, value in self.terms.items():
            if self.labels[key] in labels:
                yield value

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        """
        Returns an UFL form of the residual by summing up all the terms with the desired label.

        Sign convention: all terms are assumed to be on the right hand side of the equation: d(u)/dt = term.

        :arg label: string defining the type of terms to sum up. Currently one of
            'source'|'explicit'|'implicit'|'nonlinear'. Can be a list of multiple labels, or 'all' in which
            case all defined terms are summed.
        :arg solution: solution :class:`.Function` of the corresponding equation
        :arg solution_old: a time lagged solution :class:`.Function`
        :arg fields: a dictionary that provides all the remaining fields that the term depends on.
            The keys of the dictionary should standard field names in `field_metadata`
        :arg fields_old: Time lagged dictionary of fields
        :arg bnd_conditions: A dictionary describing boundary conditions.
            E.g. {3: {'elev_2d': Constant(1.0)}} replaces elev_2d function by a constant on boundary ID 3.
        """
        f = 0
        for term in self.select_terms(label):
            f += term.residual(solution, solution_old, fields, fields_old, bnd_conditions)
        return f

    def jacobian(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        """
        Returns an UFL form of the Jacobian by summing up all the Jacobians of the terms.

        Sign convention: all terms are assumed to be on the right hand side of the equation: d(u)/dt = term.

        :arg label: string defining the type of terms to sum up. Currently one of
            'source'|'explicit'|'implicit'|'nonlinear'. Can be a list of multiple labels, or 'all' in which
            case all defined terms are summed.
        :arg solution: solution :class:`.Function` of the corresponding equation
        :arg solution_old: a time lagged solution :class:`.Function`
        :arg fields: a dictionary that provides all the remaining fields that the term depends on.
            The keys of the dictionary should standard field names in `field_metadata`
        :arg fields_old: Time lagged dictionary of fields
        :arg bnd_conditions: A dictionary describing boundary conditions.
            E.g. {3: {'elev_2d': Constant(1.0)}} replaces elev_2d function by a constant on boundary ID 3.
        """
        f = 0
        for term in self.select_terms(label):
            # FIXME check if jacobian exists?
            f += term.jacobian(solution, solution_old, fields, fields_old, bnd_conditions)
        return f


class GOErrorEstimatorTerm(object):
    """
    Implements the component of a goal-oriented error estimator from a single term of the underlying
    equation.

    .. note::
        Sign convention as in :class:`Term`.
    """
    def __init__(self, mesh):
        """
        :arg function_space: the :class:`FunctionSpace` the solution belongs to
        """
        self.P0_2d = FunctionSpace(mesh, "DG", 0)
        self.p0test = TestFunction(self.P0_2d)
        self.p0trial = TrialFunction(self.P0_2d)

    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        # TODO: doc
        raise NotImplementedError('Must be implemented in the derived class')

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        # TODO: doc
        raise NotImplementedError('Must be implemented in the derived class')

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        # TODO: doc
        raise NotImplementedError('Must be implemented in the derived class')


class GOErrorEstimator(object):
    """
    Implements a goal-oriented error estimator, comprised of the corresponding terms from the
    underlying equation.
    """
    SUPPORTED_LABELS = frozenset(['source', 'explicit', 'implicit', 'nonlinear'])

    def __init__(self, function_space):
        """
        :arg function_space: the :class:`FunctionSpace` the solution belongs to
        """
        self.terms = OrderedDict()
        self.labels = {}
        self.function_space = function_space
        self.mesh = function_space.mesh()
        self.normal = FacetNormal(self.mesh)
        self.xyz = SpatialCoordinate(self.mesh)
        self.P0_2d = FunctionSpace(self.mesh, "DG", 0)
        self.p0test = TestFunction(self.P0_2d)
        self.p0trial = TrialFunction(self.P0_2d)

    def mass_term(self, solution, arg):
        # TODO: doc
        return self.p0test*inner(solution, arg)*dx

    def add_term(self, term, label):
        """
        Adds a term in the error estimator

        :arg term: :class:`.GOErrorEstimatorTerm` object to add_term
        :arg string label: Assign a label to the term. Valid labels are given by
            :attr:`.SUPPORTED_LABELS`.
        """
        key = term.__class__.__name__
        self.terms[key] = term
        self.label_term(key, label)

    def label_term(self, term, label):
        """
        Assings a label to the given term(s).

        :arg term: :class:`.GOErrorEstimatorTerm` object, or a tuple of terms
        :arg label: string label to assign
        """
        if isinstance(term, str):
            assert term in self.terms, 'Unknown term, add it to the equation'
            assert label in self.SUPPORTED_LABELS, 'bad label: {:}'.format(label)
            self.labels[term] = label
        else:
            for k in iter(term):
                self.label_term(k, label)

    def select_terms(self, label):
        """
        Generator function that selects terms by label(s).

        label can be a single label (e.g. 'explicit'), 'all' or a tuple of
        labels.
        """
        if isinstance(label, str):
            if label == 'all':
                labels = self.SUPPORTED_LABELS
            else:
                labels = frozenset([label])
        else:
            labels = frozenset(label)
        for key, value in self.terms.items():
            if self.labels[key] in labels:
                yield value

    def element_residual(self, label, *args):
        # TODO: doc
        residual_terms = 0
        for term in self.select_terms(label):
            residual_terms += term.element_residual(*args)
        residual = assemble(residual_terms)
        residual.rename("Element residual")
        return residual

    def inter_element_flux(self, label, *args):
        # TODO: doc
        mass_term = self.p0test*self.p0trial*dx
        flux_terms = 0
        for term in self.select_terms(label):
            flux_terms += term.inter_element_flux(*args)
        flux = Function(self.P0_2d, name="Inter-element flux terms")
        solve(mass_term == flux_terms, flux)  # TODO: Solver parameters?
        return flux

    def boundary_flux(self, label, *args):
        # TODO: doc
        mass_term = self.p0test*self.p0trial*dx
        bnd_flux_terms = 0
        for term in self.select_terms(label):
            bnd_flux_terms += term.boundary_flux(*args)
        bnd_flux = Function(self.P0_2d, name="Boundary flux terms")
        solve(mass_term == bnd_flux_terms, bnd_flux)  # TODO: Solver parameters?
        return bnd_flux
