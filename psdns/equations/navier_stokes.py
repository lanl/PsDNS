import warnings

import numpy

from psdns import *


class NavierStokes(object):
    r"""Incompressible Navier-Stokes equations

    The incompressible Navier-Stokes equations are (non-dimensionalized)

    .. math::

        u_{i,i} & = 0 \\
        u_{i,t} + u_j u_{i,j} & = - p_{,i} + \frac{1}{\mathrm{Re}} u_{i,jj}

    """
    def __init__(self, Re, **kwargs):
        """Return a Navier-Stokes equation object.

        Instantiating returns a Navier-Stokes equation object with
        Reynolds number *Re*.
        """
        super().__init__(**kwargs)

        self.Re = Re
        self.nu = 1/Re

    def rhs(self, uhat):
        r"""Compute the Navier-Stokes right-hand side

        To compute the Navier-Stokes right-hand side in spectral
        space, we first Fourier transform the equations

        .. math::

            k_i \hat{u}_i
            & = 0 \\
            \hat{u}_{i,t} + \widehat{u_j u_{i,j}}
            & = - i k_i \hat{p} - \nu k^2 \hat{u}_i

        To eliminate pressure, multiply the momentum equation by :math:`k_i`,
        and use continuity

        .. math::

            \hat{p} = i \frac{k_j}{k^2} \widehat{u_k u_{j,k}}

        Substituting back into the momentum equation, the evolution
        equation becomes

        .. math::

            \left( \frac{\partial}{\partial t} + \nu k^2 \right) \hat{u}_i
            = \left( \frac{k_i k_j}{k^2} - \delta_{ij} \right)
              \widehat{u_k u_{j,k}}

        As written, the non-linear term requires 12 backward
        (spectral-to-physical) and 3 forward transforms.  Using the
        `identity <https://en.wikipedia.org/wiki/
        Vector_calculus_identities#Cross_product_rule>`_

        .. math::

            \mathbf{A \times ( \nabla \times B )}
            = \mathbf{A \cdot \nabla B - ( A \cdot \nabla) B}

        and substituting in :math:`u_i` for both :math:`\mathbf{A}` and
        :math:`\mathbf{B}`, we have

        .. math::

            u_j u_{i,j}
            = - \boldsymbol{u} \times \boldsymbol{\omega}

        This form of the non-linear term has the advantage of requiring
        only 6 backward transforms.

        So the final form of the equations computed by
        :class:`NavierStokes` is

        .. math::

            \left( \frac{\partial}{\partial t} + \nu k^2 \right) \hat{u}_i
            = \left( \delta_{ij} - \frac{k_i k_j}{k^2} \right)
            \widehat{\left( \boldsymbol{u} \times \boldsymbol{\omega}\right)_j}
        """
        u = uhat.to_physical()
        vorticity = uhat.curl().to_physical()
        nl = numpy.cross(u, vorticity, axis=0)
        nl = PhysicalArray(uhat.grid, nl).to_spectral()
        du = numpy.einsum("ij...,j...->i...", uhat.grid.P, nl)
        du -= self.nu*uhat.grid.k2*uhat
        return du

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

    def rhs(self, uhat):
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
    def rhs(self, uhat):
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

    def rhs(self, uhat):
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
