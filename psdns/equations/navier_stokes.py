import warnings

import numpy
import scipy.special

from psdns import *
from psdns.equations import Equation

class NavierStokes(Equation):
    r"""Incompressible Navier-Stokes equations

    The incompressible Navier-Stokes equations for an incompressible
    fluid with a passive scalar are (non-dimensionalized)

    .. math::

        u_{i,i} & = 0 \\
        u_{i,t} + u_j u_{i,j} & = - p_{,i} + \nu u_{i,jj} \\
        c_{,t} + u_{j}c_{,j} & = D c_{,jj}
    """
    def __init__(self, Re, Sc=1, **kwargs):
        r"""Return a Navier-Stokes equation object.

        Instantiating returns a Navier-Stokes equation object with
        Reynolds number *Re* and Schmidt number *Sc*.

        Since passive scalars do not influence the flow, multiple scalar
        fields can be simulated simultaneously.  The initial condition
        for the :class:`NavierStokes` class should be a vector of length
        :math:`3+N`, where the first three components are the velocity
        field, and the remaining :math:`N` are an arbitrary number of
        passive scalars (:math:`N` could be zero).
        
        .. rubric:: A note on non-dimensionalization
        
        It is typical to work in non-dimensional units, however,
        internally the code converts Reynolds and Schmidt numbers to
        :math:`\nu = 1 / \mathrm{Re}` and :math:`D = 1 / \mathrm{Sc}
        \mathrm{Re}`, so that the diffusion coefficients can be set to
        zero.  Users who prefer to work in dimensional units can use any
        consistent set, and set the diffusion coefficients to the
        desired dimensional values according to these formulae.
        """
        super().__init__(**kwargs)

        self.Re = Re
        self.Sc = Sc

    def _get_Re(self):
        return 1.0/self.nu

    def _set_Re(self, Re):
        self.nu = 1.0/Re

    Re = property(_get_Re, _set_Re)
        
    def _get_Sc(self):
        return self.nu/self.D

    def _set_Sc(self, Sc):
        self.D = self.nu/Sc

    Sc = property(_get_Sc, _set_Sc)

    def rhs(self, time, uhat):
        r"""Compute the Navier-Stokes right-hand side

        Take the velocity and scalar fields in spectral space given by
        *uhat* and return a vector of the same length containing the
        right-hand side of the evolution equations.

        To compute the Navier-Stokes right-hand side in spectral
        space, we first Fourier transform the equations

        .. math::

            k_i \hat{u}_i
            & = 0 \\
            \hat{u}_{i,t} + \widehat{u_j u_{i,j}}
            & = - i k_i \hat{p} - \nu k^2 \hat{u}_i \\
            \hat{c}_{,t} + \widehat{u_{j}c_{,j}}
            & = - D k^2 \hat{c}

        To eliminate pressure, multiply the momentum equation by :math:`k_i`,
        and use continuity

        .. math::
            :label: poisson

            \hat{p} = i \frac{k_j}{k^2} \widehat{u_k u_{j,k}}

        Substituting back into the momentum equation, the evolution
        equations are

        .. math::

            \left( \frac{\partial}{\partial t} + \nu k^2 \right) \hat{u}_i
            & = - \left(  \delta_{ij} - \frac{k_i k_j}{k^2} \right)
                \widehat{u_k u_{j,k}} \\
            \left( \frac{\partial}{\partial t} + D k^2 \right) \hat{c}
            & = - \widehat{u_{j}c_{,j}}
        """
        u = uhat[:3].to_physical()
        nl = self.nonlinear(uhat, u)
        du = - numpy.einsum("ij...,j...->i...", uhat.grid.P, nl)
        du -= self.nu*uhat.grid.k2*uhat[:3]

        gradc = uhat[3:].grad().to_physical()
        dc = - numpy.einsum("j...,ij...->i...", u, gradc)
        dc = PhysicalArray(uhat.grid, dc).to_spectral()
        dc -= self.nu/self.Sc*uhat.grid.k2*uhat[3:]
        
        return numpy.concatenate((du, dc))

    def nonlinear(self, uhat, u):
        r"""Compute the non-linear term for the Navier-Stokes operator

        Return the non-linear operator from the momentume equatiom,
        :math:`\widehat{u_k u_{j,k}}`, computed psuedo-spectrally, using
        the spectral space vector of unknowns *uhat* and the physical
        space velocity field *u*.  This routines requires the velocity
        field to be passed in, since it needs to be computed in the
        :meth:`rhs` routine for use in the non-linear terms in the
        scalar equations.
        """
        gradu = uhat[:3].grad().to_physical()
        return PhysicalArray(
            uhat.grid,
            numpy.einsum("j...,ij...->i...", u, gradu)).to_spectral()

    def pressure(self, uhat):
        """Return the pressure field

        Compute the pressure field corresponding to the spectral space
        velocity field *uhat*, using the Poisson equation in,
        :eq:`poisson`.
        """
        u = uhat[:3].to_physical()
        # Call the NavierStokes method explictly, because for the
        # pressure calculation, even the RotationalNavierStokes class
        # needs to use this version of the non-linear term, otherwise
        # you get dynamic pressure instead.
        p = self.nonlinear(uhat, u).div() \
          / numpy.where(uhat.grid.k2 == 0, 1, uhat.grid.k2)
        return SpectralArray(uhat.grid, p)
    
    def press_rapid(self, uhat):
        r"""Compute the rapid pressure

        For turbulent flows, we can decompose the flow into a mean and
        fluctuating part denoted by as

        .. math::

            f = \bar{f} + f'

        For the pressure, it is straightforward to show that the
        fluctuating pressure can be further decomposed into a slow and a
        rapid part, defined by

        .. math::

            \nabla^2 p^s
            & = \frac{\partial^2}{\partial x_{i}\partial x_{j}}
            \left[
                \overline{u_{i}'u_{j}'}-u_{i}'u_{j}'
            \right] \\
            \nabla^2 p^r
            & = - 2 \frac{\partial\bar{u}_{j}}{\partial x_{i}} 
                    \frac{\partial u_{i}'}{\partial x_{j}}

        This routine takes a spectral space velocity field *uhat* and
        returns the rapid pressure, :math:`p^r`, relative to a planar
        average in the :math:`x-y` plane.
        """
        # We need z derivatives of planar averaged quantities.  The
        # most efficient way to do this would be to write specialized
        # code for doing this spectrally, taking into account the
        # domain decomposition, but for simplicty we use this less
        # efficient method.
        #
        # Note, by definition, x and y derivatives of mean velocity are
        # zero, and, by continuity, the so is the z derivative of mean w
        # velocity.  So only the following two components contribute to
        # the r.h.s.
        dudz = (1j*uhat.grid.k[2]*uhat[0]).to_physical().avg_xy()
        dvdz = (1j*uhat.grid.k[2]*uhat[1]).to_physical().avg_xy()
        dudz = uhat.grid.comm.bcast(dudz)
        dvdz = uhat.grid.comm.bcast(dvdz)
        w = uhat[2].disturbance()
        dwdx = (1j*uhat.grid.k[0]*w).to_physical()
        dwdy = (1j*uhat.grid.k[1]*w).to_physical()
        # The minus sign cancels between the r.h.s. and the -k^2 in the
        # Laplacian.
        rhs = - 2 * ( dudz * dwdx + dvdz * dwdy ).to_spectral()
        return - rhs / numpy.where(uhat.grid.k2 == 0, 1, uhat.grid.k2)

    def press_slow(self, uhat):
        r"""Compute the slow pressure

        This routine takes a spectral space velocity field *uhat* and
        returns the slow pressure, :math:`p^s`, relative to a planar
        average in the :math:`x-y` plane.
        """
        u = uhat[:3].disturbance().to_physical()
        uiuj = PhysicalArray(uhat.grid, numpy.einsum("i...,j...->ij...", u, u)).to_spectral()
        rhs = - uiuj.disturbance().div().div()
        return - rhs / numpy.where(uhat.grid.k2 == 0, 1, uhat.grid.k2)

    def taylor_green_vortex(self, grid, A=1, B=-1, C=0, a=1, b=1, c=1):
        r"""Initialize with the Taylor-Green problem

        Return a spectral space velocity field on the specified *grid*
        corresponding to the Taylor-Green vortex problem
        [Taylor1938]_.

        The remaining arguments are the coefficients in the formula

        .. math::

            u = & A \cos a x \sin b y \sin c z \\
            v = & B \sin a x \cos b y \sin c z \\
            w = & C \sin a x \sin b y \cos c z

        Note that, in order to satisfy the continuity equation, the
        coefficients must satisfy the relation

        .. math::

            A a + B b + C c = 0
        """
        assert A*a+B*b+C*c == 0, \
            "Initial condition does not satisfy continuity."

        u = PhysicalArray(grid, (3,))
        x = u.grid.x

        u[0] = A*numpy.cos(a*x[0])*numpy.sin(b*x[1])*numpy.sin(c*x[2])
        u[1] = B*numpy.sin(a*x[0])*numpy.cos(b*x[1])*numpy.sin(c*x[2])
        u[2] = C*numpy.sin(a*x[0])*numpy.sin(b*x[1])*numpy.cos(c*x[2])

        s = u.to_spectral()
        s._data = numpy.ascontiguousarray(s._data)
        return s

    @staticmethod
    def mansour(k, q2, sigma, kp):
        r"""Initial energy spectrom of [Mansour1994]_.

        Computes the initial energy spectrum of [Mansour1994]_, which is

        .. math::

            E(\kappa) 
            = \frac{q^2}{2A} \frac{\kappa^\sigma}{\kappa_p^{\sigma+1}}
            \exp \left( 
              - \frac{\sigma}{2} \left( \frac{\kappa}{\kappa_p} \right)^2
            \right)
        """
        A = ( 2 ** ( (sigma-1) / 2 ) * scipy.special.gamma( (sigma+1) / 2)
              / sigma ** ( (sigma+1) / 2 ) )
        return ( q2 / ( 2 * A ) * k ** sigma / ( kp ** (sigma+1) )
            * numpy.exp( - sigma / 2 * ( k / kp ) ** 2 ) )

    def rogallo(self, grid, energy=mansour,
                params={'q2':3, 'sigma':2, 'kp':25}):
        r"""Initialize with a specified spectrum and random phases.

        Return a :class:`~psdns.bases.SpectralArray` with
        ``shape=(3,)``, containing a velocity field on *grid* that
        satisfies continuity, has a specified *energy* spectrum, and
        random phases.

        The spectrum is specified as a function *energy*, which takes as
        arguments a wavenumber *k* and optional additional parameters
        passed as a dictionary *params*.  The default value for the
        *energy* function is the initial spectrum of [Mansour1994]_, with
        the *params* set for the case in their second figure.

        The algorithm for creating a field with the specified properties
        is that of [Rogallo1981]_.  Note that there is a missing square
        root and factor of two in [Mansour1994]_ in the equations for
        :math:`\alpha,\beta`; coefficient is

        .. math::

            \left( \frac{E(\kappa)}{2 \pi \kappa^2} \right)^{1/2}
        """
        u = SpectralArray(grid, (3,))
        k = u.grid.k
        kmag = numpy.sqrt(u.grid.k2)
        k12 = numpy.sqrt(k[0]**2+k[1]**2)

        phi = 2*numpy.pi*numpy.random.random(k.shape[1:])
        theta1 = 2*numpy.pi*numpy.random.random(k.shape[1:])
        theta2 = 2*numpy.pi*numpy.random.random(k.shape[1:])
        alpha = numpy.exp(1j*theta1)*numpy.cos(phi)
        beta = numpy.exp(1j*theta2)*numpy.sin(phi)

        u[0] = numpy.where(k12 == 0, alpha, (alpha*kmag*k[1] + beta*k[0]*k[2])/(kmag*k12))
        u[1] = numpy.where(k12 == 0, beta, (- alpha*kmag*k[0] + beta*k[1]*k[2])/(kmag*k12))
        u[2] = numpy.where(k12 == 0, 0, -beta*k12/kmag)
        u *= numpy.sqrt(numpy.prod(u.grid.dk)*energy(kmag, **params)/(2*numpy.pi*u.grid.k2))
        u[:,0,0,0] = 0

        return u

    def shear(self, grid, d, modes=[ ( 1e-1, 1, 0 ), ( 1e-2, 0, 1 ) ]):
        r"""Initialize a perturbed shear layer.

        This method returns an initial condition for a shear layer
        with superimposed perturbations.  The shear layer is defined
        by an error-function profile of width *d*, and, to satisfy the
        periodic boundary conditions, is really an infinite series of
        alternating-direction shear-layers at locations :math:`z=0,
        L_z/2, L, \ldots`, where :math:`L_z` is the extent of the
        domain in the vertial direction.  This follows the approach in
        [Sharan2019]_.

        There are various ways of introducting perturbations, of
        varying levels of physical fidelity.  For small amplitute
        perturbations, one could introduce disturbances obtained from
        linear stability theory.  In order to simplify things, this
        routine uses an analytic expression to provide a disturbance
        that looks similar to on from linear theory.  We use a
        disturbance of the form:

        .. math::

          u'(x, y, z)
          =  A f' \left( \frac{z}{\gamma} \right)
          \sin ( \alpha x - \phi_x ) \cos ( \beta y - \phi_y )

          v'(x, y, z)
          =  A f' \left( \frac{z}{\gamma} \right)
          \sin ( \alpha x - \phi_x ) \cos ( \beta y - \phi_y )

          w'(x, y, z)
          = - A f \left( \frac{z}{\gamma} \right)
          \sin ( \alpha x - \phi_x ) \cos ( \beta y - \phi_y )

        This will produce a disturbance the satisfies continuity
        provided that :math:`\alpha + \beta = \gamma^{-1}`.  For the
        disturbance function, we choose a Gaussian, :math:`f(\eta) =
        \exp ( - \eta^2 )`.

        In addition to the shear-layer thickness, users provide
        *modes*, a list of 3-tuples, each of which is of the form
        :math:`(A, \alpha, \beta)`.  A default option which is
        observed to lead to a reasonable transition is included.

        .. note::

          In order for the initial conditions to satisfy periodicity
          in the normal planes, :math:`\alpha, \beta` and the box size
          must be chosen such that :math:`\alpha L_x, \beta L_y` are
          always multiples of :math:`2 \pi`.  The code does not
          enforce this condition.
        """
        u = PhysicalArray(grid, (3,))
        u[0] = scipy.special.erf(grid.x[2]/d) \
          - scipy.special.erf((grid.x[2] - grid.box_size[2]/2) / d) \
          + scipy.special.erf((grid.x[2] - grid.box_size[2]) / d)
        z = grid.x[2] - grid.box_size[2] / 2
        for A, a, b in modes:
            c = 1 / ( a + b )
            u[0] = u[0] - 2 * A * z / c * numpy.exp(-(z/c)**2) \
              * numpy.sin(a*grid.x[0]) * numpy.cos(b*grid.x[1])
            u[1] = u[1] - 2 * A * z / c * numpy.exp(-(z/c)**2) \
              * numpy.cos(a*grid.x[0]) * numpy.sin(b*grid.x[1])
            u[2] = u[2] - A * numpy.exp(-(z/c)**2) \
              * numpy.cos(a*grid.x[0]) * numpy.cos(b*grid.x[1])
        s = u.to_spectral()
        s._data = numpy.ascontiguousarray(s._data)
        return s


class RotationalNavierStokes(NavierStokes):
    r"""Incompressible Navier-Stokes in rotational form

    The computation of the non-linear term in the standard form, as
    implemented in :meth:`NavierStokes.nonlinear`, 

    This incompressible Navier-Stokes can be written in an alternate
    form, termed rotational form, requires 12 backward
    (spectral-to-physical) and 3 forward transforms.  Using the
    `identity <https://en.wikipedia.org/wiki/
    Vector_calculus_identities#Cross_product_rule>`_        
    
    .. math::
    
        \mathbf{A \times ( \nabla \times B )}
        = \mathbf{A \cdot \nabla B - ( A \cdot \nabla) B}

    and substituting in :math:`u_i` for both :math:`\mathbf{A}` and
    :math:`\mathbf{B}`, we can re-write the non-linear term as,

    .. math::

        \mathbf{( u \cdot \nabla) u}
        = \frac{1}{2} \nabla u^2 - \boldsymbol{u} \times \boldsymbol{\omega}

    The momentum equation then can be written as

    .. math::

        \frac{\partial u_i}{\partial t} -  \boldsymbol{u} \times \boldsymbol{\omega}
        = - \frac{\partial}{\partial x_i} \left[ p + \frac{1}{2} u^2 \right] 
        + \nu \nabla^2 u_i

    This form of the non-linear term has the advantage of requiring only
    6 backward transforms.  The pressure term (in brackets) is now
    the dynamic pressure.  The Poisson equation for the dynamic pressure
    is (in Fourier space)

    .. math::

        \widehat{\left[ p + \frac{1}{2} u^2 \right]}
        = - i \frac{k_j}{k^2} \widehat{( \boldsymbol{u} \times \boldsymbol{\omega})}_j
    
    Eliminating dynamic pressure from the momentum equation, the final
    equation to be solved is

    .. math::

        \left( \frac{\partial}{\partial t} + \nu k^2 \right) \hat{u}_i
        = \left( \delta_{ij} - \frac{k_i k_j}{k^2} \right)
            \widehat{\left( \boldsymbol{u} \times \boldsymbol{\omega}\right)_j}
    """
    def nonlinear(self, uhat, u):
        r"""Compute the rotational Navier-Stokes non-linear term

        Returns the non-linear term :math:`\boldsymbol{u} \times
        \boldsymbol{\omega}` with the vorticity computed from spectral
        velocity fields in the first three elements of *uhat* and the
        velocity from the physical space velocity *u*.
        """
        vorticity = uhat[:3].curl().to_physical()
        nl = - numpy.cross(u, vorticity, axis=0)
        nl = PhysicalArray(uhat.grid, nl).to_spectral()
        return nl

    def pressure(self, uhat):
        """Return the pressure field

        Compute the pressure field corresponding to the spectral space
        velocity field *uhat*, using the Poisson equation in,
        :eq:`poisson`.  Note that the pressure cannot be computed using
        the rotational non-linear term, as that would yield the dynamic
        pressure.
        """
        return NavierStokes.pressure(self, uhat)
    

class Boussinesq(RotationalNavierStokes):
    r"""The Navier-Stokes equation with Boussinesq approximation for buoyancy.

    The incompressible Navier-Stokes equations, with a scalar
    transport equation and the Boussinesq approximation for buoyancy
    are:

    .. math::

        u_{i,i} & = 0 \\
        u_{i,t} + u_{j} u_{i,j} 
        & = - p_{,i} + \nu u_{i,jj} + c g_i \\
        c_{,t} + u_{j}c_{,j} & = D c_{,jj}

    Denoting the non-linear term in the momentume equation
    :math:`\mathrm{NL}`, we can eliminate pressure by Fourier
    transforming, and writing the pressure Poisson equation as

    .. math::

        \hat{p}
        & = i \frac{k_j}{k^2} \left( \widehat{\mathrm{NL}}_j - \hat{c} g_{j} \right) \\

    so the momentum equation to be solved is
    
    .. math::

        \left( \frac{\partial}{\partial t} +  \nu k^2 \right) \hat{u}_i
        = - \left( \delta_{ij} -\frac{k_i k_j}{k^2} \right) 
            \left( \widehat{\mathrm{NL}}_j - \hat{c} g_j \right) 
    """
    def __init__(self, g=-1, **kwargs):
        """Return a Boussinesq equation object.

        Intantiating returns a Boussinsesq equation object with
        Reynolds number *Re* and Schmidt number *Sc*.
        """
        super().__init__(**kwargs)

        self.g = g

    def nonlinear(self, uhat, u):
        r"""Compute the Boussinesq non-linear term.

        The numerical implementation is the same as for
        :meth:`psdns.equations.navier_stokes.NavierStokes.rhs` execpt
        for the additional bouyancy term.  As a result the momentum
        equation becomes

        .. math::

            \left(
              \frac{\partial}{\partial t} + \nu k^{2}
            \right) \hat{u}_{i}
            = \left( \frac{k_{i}k_{j}}{k^{2}} - \delta_{ij} \right)
            \left( \widehat{u_{k}u_{j,k}} - \hat{c} g_{j} \right)
        """
        nl = super().nonlinear(uhat, u)
        nl[2] -= self.g*uhat[3]
        return nl

    def pressure(self, uhat):
        u = uhat[:3].to_physical()
        nl = NavierStokes.nonlinear(self, uhat, u)
        nl[2] -= self.g*uhat[3]
        p = nl.div() \
          / numpy.where(uhat.grid.k2 == 0, 1, uhat.grid.k2)
        return SpectralArray(uhat.grid, p)

    def press_buoyant(self, uhat):
        p = - 1j * uhat.grid.k[2] * uhat[3].disturbance() * self.g \
          / numpy.where(uhat.grid.k2 == 0, 1, uhat.grid.k2)
        return SpectralArray(uhat.grid, p)
    
    def ic(self, grid):
        u = PhysicalArray(grid, (4,))
        x = u.grid.x
        eta = x[2] - 4*numpy.pi
        A = 0.1
        k = 1
        kx = ky = 1
        u[3] = scipy.special.erf(eta/0.1+0.1*numpy.cos(x[0])*numpy.cos(x[1]))-scipy.special.erf(x[2]/0.1)-scipy.special.erf((x[2]-8*numpy.pi)/0.1)
        s = u.to_spectral()
        s._data = numpy.ascontiguousarray(s._data)
        return s

    def perturbed_interface(self, grid, z, delta1, delta2, profile=scipy.special.erf):
        """Creates a perturbed interface.

        *z* is the pertubation function.  *delta1* is the width of the
        unstable interface, and *delta2* is the width of the stable 
        interface.
        """
        u = PhysicalArray(grid, (4,))
        x = u.grid.x
        x1 = u.grid.box_size[2]/2
        x2 = u.grid.box_size[2]
        u[3] = (
            profile((x[2] - x1 - z[:,:,numpy.newaxis])/delta1)
            - profile(x[2]/delta2)
            - profile((x[2] - x2)/delta2)
            )
        s = u.to_spectral()
        s._data = numpy.ascontiguousarray(s._data)
        return s

    def band(self, grid, kmin, kmax, seed=123):
        if seed == None:
            warnings.warn(
                "A seed of None for Boussinesq.band() will result "
                "in inconsistent initialization across MPI ranks."
                )
        # Check kmax fits on the grid!
        x = grid.x[:2,:,:,0]
        z = numpy.zeros(shape=x[0].shape)
        # Since the loop will execute identically on all ranks, rng will
        # generate the same random numbers.
        rng = numpy.random.default_rng(seed)
        for n in range(kmax):
            for m in range(kmax):
                k = numpy.sqrt(n**2 + m**2)
                if k >= kmin and k <= kmax:
                    z += (
                        numpy.cos(2*numpy.pi*(n*x[0]/grid.box_size[0]+rng.random()))
                        *numpy.cos(2*numpy.pi*(m*x[1]/grid.box_size[1]+rng.random()))
                        )
        return z


class SimplifiedSmagorinsky(NavierStokes):
    r"""A simplified Smagorinsky-type model.

    The Smagorinsky LES model is obtained by replacing the molecular
    viscosity in the Navier-Stokes equations with the sum of a molecular
    and an eddy-viscosity.  The full formulation (implemented in
    :class:`Smagorinsky`) adds additional non-linearities to the
    equations.  However, two simplification can be used to create a
    version of the Smagorinsky model which can be extremely efficiently
    implemented in the pseudo-spectral framework.  The first is to use a
    spatially averaged eddy-viscosity, so that there are no
    eddy-viscosity derivative terms.  The second is to use the magnitude
    of the curl, rather than the velocity gradient tensor, to compute
    the velocity scale.  The eddy-viscosity is then

    .. math::

        \nu_T = (C_S \Delta)^2 \sqrt{<\omega_i \omega_i>}

    where the brackets are a spatial average over the entire periodic
    box.

    The model can then be implemented simply by adding the
    eddy-viscosity to the molecular viscosity.
    """
    def __init__(self, Cs=0.17, **kwargs):
        """Returns a simplified Smagorinsky equation object.

        Intantiating returns a simplified Smagorinsky equation object
        with Smagorinsky coefficient *Cs*.  Remaining arguments are
        passed to the :class:`NavierStokes`.
        """
        super().__init__(**kwargs)
        self.Cs = Cs

    def rhs(self, time, uhat):
        u = uhat.to_physical()
        vorticity = uhat.curl()
        nu_t = (self.Cs*uhat.grid.dx)**2 \
            * numpy.sqrt(numpy.sum(vorticity.norm()))
        nl = numpy.cross(u, vorticity.to_physical(), axis=0)
        nl = PhysicalArray(uhat.grid, nl).to_spectral()
        du = numpy.einsum("ij...,j...->i...", uhat.grid.P, nl)
        du -= (self.nu+nu_t)*uhat.grid.k2*uhat
        return du


class Smagorinsky(SimplifiedSmagorinsky):
    r"""The standard Smagorinsky model.

    With the addition of a linear eddy-viscosity model, the filtered
    momentum equation becomes (in the following assume :math:`u_i,p`
    are filtered quantities)

    .. math::

        u_{i,t} + u_j u_{i,j}
        = -p_{,i} + \left( \nu + \nu_T u_{i,j} \right)_{,j}

    Transforming to spectral space,

    .. math::

        \hat{u}_{i,t} + \widehat{u_j u_{i,j}}
        = - i k_i \hat{p} - \nu k^2 \hat{u}_i
        + i k_j \widehat{\nu_T u_{i,j}}

    To eliminate pressure, multiply the momentum equation by
    :math:`k_i`, and use continuity

    .. math::

        \hat{p}
        = \frac{k_j}{k^2}
        \left( i \widehat{u_k u_{j,k}} + k_k \widehat{\nu_T u_{j,k}} \right)

    The momentum equation becomes

    .. math::

        \left( \frac{d}{d t} + \nu k^2 \right) \hat{u}_i
        = \left( \delta_{ij} - \frac{k_i k_j}{k^2} \right)
        \left( i k_k \widehat{\nu_T u_{j,k}} - \widehat{u_k u_{j,k}} \right)

    Note that with this formulation, re-writing the non-linear term in
    terms of the vorticity is no longer beneficial, since the full
    velocity gradient tensor is still required for the eddy-viscosity
    term.

    The eddy-viscosity is given by

    .. math::

        \nu_T = \left( C_s \Delta \right)^2 \sqrt{S_{ij} S_{ij}}

    The constructor is the same as for the
    :class:`SimplifiedSmagorinsky` class.
    """
    def rhs(self, time, uhat):
        dx = numpy.sqrt(numpy.sum(uhat.grid.dx**2)/3)
        u = uhat.to_physical()
        gradu = uhat.grad().to_physical()
        Sij = ((gradu + gradu.transpose(1, 0, 2, 3, 4))/2)
        nu_t = (self.Cs*dx)**2*numpy.sqrt(numpy.sum(Sij*Sij, axis=(0, 1)))
        nl1 = numpy.einsum(
            "k...,jk...->j...",
            uhat.grid.k,
            (nu_t*gradu).to_spectral()
            )
        nl2 = numpy.einsum("k...,jk...->j...", u, gradu)
        nl2 = PhysicalArray(uhat.grid, nl2).to_spectral()
        nl = 1j*nl1 - nl2
        du = numpy.einsum("ij...,j...->i...", uhat.grid.P, nl)
        du -= self.nu*uhat.grid.k2*uhat
        return du


class KEpsilon(NavierStokes):
    r"""A two-equation :math:`K-\varepsilon` model.

    The :math:`K-\varepsilon` model is a two-equation RANS model, with
    new equations for the turbulent kinetic energy and the energy
    dissipation rate, :math:`K` and :math:`\varepsilon`, respectively.

    The :math:`K-\varepsilon` model equations are

    .. math::

        K_{,t} + u_j k_{,j}
        & = \mathcal{P} - \varepsilon
        + \left( \left( \nu + \nu_T \right) K_{i,j} \right)_{,j} \\
        \varepsilon_{,t} + u_j \varepsilon_{,j}
        &= \frac{\varepsilon}{K}
        \left(
          C_{\varepsilon1} \mathcal{P}
          - C_{\varepsilon2} \varepsilon
        \right)
        + \left( \left( \nu + \nu_T \right) \varepsilon_{i,j} \right)_{,j}

    where

    .. math::

        \mathcal{P} = \nu_T S_{ij} S_{ij}

    The momentum equation is the same as used for the
    :class:`Smagorinsky` model, except that the eddy-viscosity is

    .. math::

        \nu_T = C_\mu \frac{K^2}{\varepsilon}
    """
    def __init__(
            self,
            Ce1=1.44, Ce2=1.92, Cmu=0.09,
            Pr_k=1.0, Pr_e=1.4,
            clip=False,
            **kwargs
            ):
        r"""Return a :math:`K-\varepsilon` equation object.

        The parameters *Ce1*, *Ce2*, *Cmu*, *Pr_k*, and *Pr_e*, are the
        model coefficients.  If *clip* is `True`, then :math:`K` and
        :math:`\varepsilon` are set to a small number whenever they
        become negative. All other parameters are passed back to the
        :class:`NavierStokes` constructor.
        """
        super().__init__(**kwargs)
        self.Ce1 = Ce1
        self.Ce2 = Ce2
        self.Cmu = Cmu
        self.Pr_k = Pr_k
        self.Pr_e = Pr_e
        self.clip = clip

    def rhs(self, time, uhat):
        r"""Compute the right-hand side for the :math:`k-\varepsilon` equations

        The :math:`K-\varepsilon` equations can be treated as
        passive scalars with the appropriate source terms.

        The general equation for a scalar with a forcing term is

        .. math::

            \phi_{,t} + u_i \phi_{,i}
            = F + \nu \phi_{,ii} + \left( \nu_T \phi_{,i} \right)_{,i}

        or, in spectral space,

        .. math::

            \left( \frac{\partial}{\partial t} + k^2 \nu \right) \hat{\phi}
            = \hat{F} - \widehat{u_i \phi_{,i}}
            + i k_i \widehat{\nu_T \phi_{,i}}
        """
        # Clipping
        K = uhat[3].to_physical()
        epsilon = uhat[4].to_physical()
        if self.clip and numpy.amin(K) <= 0:
            warnings.warn("negative K: clipping to 1e-12")
            K = K.clip(1e-12)
            uhat[3] = K.to_spectral()
        if self.clip and numpy.amin(epsilon) <= 0:
            warnings.warn("negative epsilon: clipping to 1e-12")
            K = K.clip(1e-12)
            epsilon = epsilon.clip(1e-12)
            uhat[3] = K.to_spectral()
            uhat[4] = epsilon.to_spectral()

        nu_t = self.Cmu*K**2/epsilon

        # Momentum equation
        u = uhat[:3].to_physical()
        gradu = uhat[:3].grad().to_physical()
        nl1 = numpy.einsum(
            "k...,jk...->j...",
            uhat.grid.k,
            (nu_t*gradu).to_spectral()
            )
        nl2 = numpy.einsum("k...,jk...->j...", u, gradu)
        nl2 = PhysicalArray(uhat.grid, nl2).to_spectral()
        nl = 1j*nl1 - nl2
        du = numpy.einsum("ij...,j...->i...", uhat.grid.P, nl)
        du -= self.nu*uhat.grid.k2*uhat[:3]

        # Turbulent prodution
        Sij = ((gradu + gradu.transpose(1, 0, 2, 3, 4))/2)
        P = 2*nu_t*numpy.sum(Sij*Sij, axis=(0, 1))

        # K equation
        gradk = uhat[3].grad().to_physical()
        dk = P - epsilon - numpy.einsum("i...,i...", u, gradk)
        dk = PhysicalArray(uhat.grid, dk).to_spectral()
        dk += 1j*numpy.einsum(
            "i...,i...",
            uhat.grid.k,
            (nu_t/self.Pr_k*gradk).to_spectral()
            )
        dk -= self.nu*uhat.grid.k2*uhat[3]

        # epsilon equation
        grade = uhat[4].grad().to_physical()
        de = epsilon/K*(self.Ce1*P - self.Ce2*epsilon) \
            - numpy.einsum("i...,i...", u, grade)
        de = PhysicalArray(uhat.grid, de).to_spectral()
        de += 1j*numpy.einsum(
            "i...,i...",
            uhat.grid.k,
            (nu_t/self.Pr_e*grade).to_spectral()
            )
        de -= self.nu*uhat.grid.k2*uhat[4]

        return numpy.concatenate(
            (
                du,
                dk[numpy.newaxis, ...],
                de[numpy.newaxis, ...],
            )
        )
