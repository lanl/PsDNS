import unittest


import matplotlib
matplotlib.use('PDF')
import matplotlib.pylab as plt
import numpy
from numpy import testing as nptest


from psdns.diagnostics import Diagnostics
from psdns.integrators import RungeKutta
from psdns.solvers import NavierStokes, TaylorGreenIC


brachet_data = numpy.array([
    [ 0.04939965322977513, 0.007515479870643941 ],
    [ 0.18744323150022435, 0.0074550578313179825 ],
    [ 0.32539614217164736, 0.0074492781316720935 ],
    [ 0.4633274653194921, 0.00745650851290241 ],
    [ 0.6011853908871727, 0.00750797316911183 ],
    [ 0.7390433164548531, 0.0075594378253212476 ],
    [ 0.8768235269376534, 0.0076577387726850105 ],
    [ 1.0145562448685834, 0.007784661897976429 ],
    [ 1.1522501052570728, 0.007935003168845017 ],
    [ 1.2899094256078376, 0.008106160569115537 ],
    [ 1.4275428409303093, 0.008292930066437505 ],
    [ 1.5651503512244873, 0.00849531166081092 ],
    [ 1.702680146433786, 0.008744529546338682 ],
    [ 1.8402444816808088, 0.00897293130246451 ],
    [ 1.977752689366529, 0.009235159268868476 ],
    [ 2.1152522620428185, 0.009502591267622925 ],
    [ 2.2527129771766683, 0.009793441411954545 ],
    [ 2.390208232348242, 0.010063475426884237 ],
    [ 2.5276991700151004, 0.010336111457989168 ],
    [ 2.665194425186675, 0.010606145472918858 ],
    [ 2.80270695037711, 0.010865771423147583 ],
    [ 2.9402367455864082, 0.011114989308675343 ],
    [ 3.0777751758051375, 0.01135900316185262 ],
    [ 3.2153222410332987, 0.011597812982679413 ],
    [ 3.352903846299184, 0.011815806674104277 ],
    [ 3.490507039088646, 0.012020790284652935 ],
    [ 3.628166359439411, 0.012191947684923455 ],
    [ 3.7658472673137533, 0.012350095004317767 ],
    [ 3.9035583977211052, 0.012490028210485392 ],
    [ 4.032691185946739, 0.012603977234106566 ],
    [ 4.165285533578067, 0.012710105817519631 ],
    [ 4.303074379070299, 0.012803202732532909 ],
    [ 4.440897764600256, 0.01287548351814426 ],
    [ 4.578747055158504, 0.012932152206704161 ],
    [ 4.716630885754478, 0.012968004765862132 ],
    [ 4.840763463831408, 0.012977894729997233 ],
    [ 4.992502167059599, 0.012977261495972284 ],
    [ 5.130498252778178, 0.01294546163457398 ],
    [ 5.268507291010902, 0.012905855724649952 ],
    [ 5.4065638217954985, 0.012837627636798271 ],
    [ 5.54466352762725, 0.012743379387194177 ],
    [ 5.682811788433499, 0.01261986867295962 ],
    [ 5.820940654375634, 0.012508046596831643 ],
    [ 5.959143980320558, 0.012351349959021755 ],
    [ 6.097377528798492, 0.012176439207985178 ],
    [ 6.235632664800003, 0.011988518376072396 ],
    [ 6.373918023334524, 0.011782383430932925 ],
    [ 6.512242239411485, 0.01155283034021628 ],
    [ 6.650566455488446, 0.011323277249499636 ],
    [ 6.788899306574836, 0.01108852012643251 ],
    [ 6.913940708092023, 0.010864076610406464 ],
    [ 7.0656211363089225, 0.01058517967002012 ],
    [ 7.204031702480193, 0.010303586255798651 ],
    [ 7.342420681127888, 0.010035002922453387 ],
    [ 7.480723309681271, 0.00981845991261295 ],
    [ 7.5915752432531525, 0.009518767518623872 ],
    [ 7.695646671297676, 0.009305459789123272 ],
    [ 7.840695265340693, 0.009023348244196287 ],
    [ 7.979088561493103, 0.008752162894675784 ],
    [ 8.117516397683236, 0.008460161415753346 ],
    [ 8.255901058826215, 0.008194180098583324 ],
    [ 8.394298672483341, 0.007920392732887579 ],
    [ 8.532687651131031, 0.007651809399542315 ],
    [ 8.671055042255148, 0.0073962361470732586 ],
    [ 8.809400845855688, 0.007153672975480408 ],
    [ 8.947763919475086, 0.0069007017391865925 ],
    [ 9.086096770561479, 0.006665944616119466 ],
    [ 9.224429621647872, 0.00643118749305234 ],
    [ 9.362771107743693, 0.00619122633763473 ],
    [ 9.501086688811224, 0.005966877279268567 ],
    [ 9.639397952374036, 0.0057451302370776485 ],
    [ 9.77767035839441, 0.005546801340463902 ],
    [ 9.902109479306144, 0.005371948156156867 ], ] ).T


class Equations(NavierStokes, TaylorGreenIC):
    pass


class TestDiagnostics(Diagnostics):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dumps = []
    
    def diagnostic(self, time, equations, uhat):
        eps = [ (1j*uhat.grid.k[i]*uhat[j]).to_physical().norm()
                for i in range(3) for j in range(3) ]
        self.dumps.append( [ time, equations.nu*sum(eps) ] )


class TestDNS(unittest.TestCase):
    def test_Brachet_Re100(self):
        """Comparision of DNS to Brachet (1983) data, Re=100

        This is a low-Reynolds number case run at coarse resolution
        and high tolerance.  It is at about the limit of the
        acceptable run time for a unit test (currently about 18
        seconds on a 2.8 GHz Intel i7).

        A currently unexplained phenomenon: if the scalings for the
        truncated modes are commented out, then this test passes,
        although both the "round-trip" and comparision to mpi4py-fft
        tests fail.  If they are uncommented, the latter tests pass,
        but this test requires N=2**5 to pass.
        """
        solver = RungeKutta(
            dt=0.01,
            tfinal=10.0,
            equations=Equations(
                Re=100,
                N=21,
                padding=1.55,
            ),
            diagnostics=[
                TestDiagnostics(tdump=0.1)
                ]
        )
        solver.run()
        output = numpy.array(solver.diagnostics_list[0].dumps).T
        plt.plot(brachet_data[0], brachet_data[1], label="Brachet (1993)")
        plt.plot(output[0], output[1], label="This code")
        plt.title("Comparision to published data")
        plt.xlabel("Time")
        plt.ylabel("Dissiapation Rate")
        plt.legend()
        plt.savefig("Brachet_Re100.pdf")
        nptest.assert_allclose(
            output[1],
            numpy.interp(output[0], brachet_data[0], brachet_data[1]),
            rtol=0.02, atol=0
            )