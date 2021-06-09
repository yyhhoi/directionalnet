# Linear-circular regression
#
# Method published in Kempter, R., Leibold, C., Buzsáki, G., Diba, K., & Schmidt, R. (2012).
# Quantifying circular – linear associations : Hippocampal phase precession. 207, 113–124.
# https://doi.org/10.1016/j.jneumeth.2012.03.007
#
# This python script is first implemented by Dr. Dustin Fetterhoff
# Minor modifications in this version by Yuk-Hoi Yiu

import numpy
from scipy.optimize import fminbound
from scipy.special import erf

def rcc(lindat0, phidat0, abound=(-2., 2.), display=False):
    """
    Calculate slope and correlation value on circular data (phidat).
    Parameters
    ----------
    lindat0 : list or ndarray
    phidat0 : list or ndarray
        Phases. Expected in radians.
    abound : list, optional
        Bounds of slope. Default [-2., 2.].
    display : bool, optional
        Display figure with fitting parameters. Default False.
    Returns
    -------
    results_dict : dict
        Dictionary containing the keys: 'rho', 'p', 'R', 'aopt', 'phi0'
    """

    assert len(lindat0) == len(phidat0)  # Size of x, y data must be the same
    results_dict = {}  # return dictionary

    if len(lindat0) < 2:
        for i in ('rho', 'p', 'R', 'aopt', 'phi0'):
            results_dict[i] = numpy.nan
        return results_dict

    else:

        # copy input array because we will do changes to them below
        lindat = numpy.copy(lindat0)
        phidat = numpy.copy(phidat0)

        global aopts, Rs
        Rs = []
        aopts = []

        phidat = numpy.float_(phidat)  # make lindat to floats
        lindat = numpy.float_(lindat)  # make lindat to floats
        # lindat = lindat/lindat.max()    # normalize lindat

        # starting points of maximization
        Nrep = 20
        da = abs(abound[1] - abound[0]) / Nrep
        astart = min(abound)

        Rfunc = lambda a: -numpy.absolute(numpy.mean(numpy.exp(1j * (phidat - 2. * numpy.pi * a * lindat))))

        aopt = numpy.nan
        R = -10 ** 10
        for n in range(Nrep):
            a0 = astart + n * da
            a1 = astart + (n + 1) * da

            returnValues = fminbound(Rfunc, a0, a1, full_output=True, disp=1)  # xtol=1e-10
            # print returnValues
            # atmp = returnValues[0][0]
            atmp = returnValues[0]
            rtmp = returnValues[1]

            if display:
                aopts.append(atmp)
                Rs.append(-rtmp)

            if -rtmp > R:
                #   print rtmp, R, aopt, atmp
                R = -rtmp
                aopt = atmp

        # phase offset, like the intercept
        v = numpy.mean(numpy.exp(1j * (phidat - 2 * numpy.pi * aopt * lindat)))
        phi0 = numpy.mod(numpy.angle(v), 2 * numpy.pi)

        theta = numpy.angle(numpy.exp(2 * numpy.pi * 1j * numpy.abs(aopt) * lindat))

        thmean = numpy.angle(numpy.sum(numpy.exp(1j * theta)))  # Return counterclockwise angle
        phmean = numpy.angle(numpy.sum(numpy.exp(1j * phidat)))

        sthe = numpy.sin(theta - thmean)
        sphi = numpy.sin(phidat - phmean)

        c12 = numpy.sum(sthe * sphi)
        c11 = numpy.sum(sthe * sthe)
        c22 = numpy.sum(sphi * sphi)

        # numpy.seterr('raise')  # For debugging of any runtime warning
        rho = c12 / numpy.sqrt(c11 * c22)  # C-L Correlation coefficient
        lam22 = numpy.mean(sthe ** 2. * sphi ** 2)
        lam20 = numpy.mean(sphi ** 2)
        lam02 = numpy.mean(sthe ** 2)
        tval = rho * numpy.sqrt(lindat.size * lam20 * lam02 / lam22)
        p = 1 - erf(numpy.abs(tval) / numpy.sqrt(2))  # Is this the p - value of rho? Different from slope?


        results_dict = {}
        for i in ('rho', 'p', 'R', 'aopt', 'phi0'):
            results_dict[i] = locals()[i]

        return results_dict