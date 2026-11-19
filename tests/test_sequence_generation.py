import math
import unittest

from DTpy import Constant, M2SE, PGSE, Scanner, STEAM
from DTpy.sequence.sequence_helper import solve_trapezoidal_sequence
from DTpy.spin import Proton
from DTpy.sequence.steam import InvalidSequenceException


class SequenceHelperTest(unittest.TestCase):
    def test_trapezoidal_sequence_round_trip(self):
        gamma = Proton.gamma
        gradient = 40e-3
        delta = 50e-3
        epsilon = gradient / 60
        eta = 5e-3

        bvalue = solve_trapezoidal_sequence(
            b=None,
            g=gamma,
            G=gradient,
            D=delta,
            n=eta,
            e=epsilon,
        )

        solved_eta = solve_trapezoidal_sequence(
            b=bvalue,
            g=gamma,
            G=gradient,
            D=delta,
            e=epsilon,
        )

        self.assertTrue(math.isclose(solved_eta, eta, rel_tol=1e-9, abs_tol=1e-12))


class SteamSequenceTest(unittest.TestCase):
    def test_generates_valid_steam_waveform(self):
        scanner = Scanner(gradientStrength=40e-3, slewRate=60)
        sequence = STEAM(bvalue=200e6, Delta=50e-3, scanner=scanner)

        self.assertEqual(len(sequence.timepoints), 14)
        self.assertEqual(len(sequence.gradients), 14)
        self.assertEqual(sequence.timepoints, sequence.t)
        self.assertEqual(sequence.gradients, sequence.G)
        self.assertEqual(sequence.Delta, 50e-3)
        self.assertGreater(sequence.TE, 0)
        self.assertGreater(sequence.TM, 0)
        self.assertLessEqual(max(abs(gradient) for gradient in sequence.gradients), scanner.gradientStrength)
        self.assertTrue(
            all(t1 >= t0 for t0, t1 in zip(sequence.timepoints, sequence.timepoints[1:]))
        )

    def test_rejects_invalid_steam_sequence(self):
        scanner = Scanner(gradientStrength=40e-3, slewRate=60)

        with self.assertRaises(InvalidSequenceException):
            STEAM(bvalue=200e6, Delta=1e-3, scanner=scanner)


class ScannerDefaultsTest(unittest.TestCase):
    def test_scanner_defaults_are_usable(self):
        scanner = Scanner()

        self.assertEqual(scanner.gradientStrength, 40e-3)
        self.assertEqual(scanner.slewRate, 60)


class SequenceStubTest(unittest.TestCase):
    def test_constant_generates_discrete_sequence(self):
        sequence = Constant(NT=3, dt=0.5, gradient=2)

        self.assertEqual(sequence.dt, [0.5, 0.5, 0.5])
        self.assertEqual(sequence.grad, [2, 2, 2])
        self.assertEqual(sequence.timepoints, [0.0, 0.5, 1.0])
        self.assertEqual(sequence.gradients, [2, 2, 2])

    def test_incomplete_generators_raise_clear_errors(self):
        for generator in (PGSE, M2SE):
            with self.subTest(generator=generator.__name__):
                with self.assertRaises(NotImplementedError):
                    generator()


if __name__ == "__main__":
    unittest.main()
