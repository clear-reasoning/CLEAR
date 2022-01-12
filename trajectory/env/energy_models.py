import math
import numpy as np


GRAMS_PER_SEC_TO_GALS_PER_HOUR = {
    'diesel': 1.119,  # 1.119 gal/hr = 1g/s
    'gasoline': 1.268,  # 1.268 gal/hr = 1g/s
}

GRAMS_TO_JOULES = {
    'diesel': 42470,
    'gasoline': 42360,
}

# COMPACT_SEDAN_COEFFS = {'p1': 0.07514808209771151, 'C0': 0.1941159506656051, 'C1': 0.01095647176178264, 'C2': 0, 'C3': 3.380641817681487e-05, 'p0': 0, 'p2': 0.0006316628238369222, 'q0': 0, 'q1': 0.01081333078118443, 'beta0': 0, 'b1': 8148.182416621619, 'b2': 3.3744524742768203, 'b3': 13.261656887955258, 'b4': -0.08420839739445923, 'b5': 4.136687639748241, 'ver': '1.0', 'mass': 1450, 'conversion': 33430.0, 'fuel_type': 'gasoline'}
# MIDSIZE_SEDAN_COEFFS = {'p1': 0.06237034371868556, 'C0': 0.26765108208781063, 'C1': 0.012474568173697364, 'C2': 0, 'C3': 2.863176152397707e-05, 'p0': 0.024237042640865628, 'p2': 0.0019643226534758955, 'q0': 0, 'q1': 0.01889090942571041, 'beta0': 0, 'b1': 8641.676403691768, 'b2': 3.4999999999832223, 'b3': 12.72850355369195, 'b4': -0.09185793598489518, 'b5': 4.775232979773187, 'ver': '1.0', 'mass': 1743, 'conversion': 33430.0, 'fuel_type': 'gasoline'}
RAV4_2019_COEFFS = {'beta0': 0.013111753095302022, 'vc': 5.98, 'p1': 0.047436831067050676, 'C0': 0.14631964767035743, 'C1': 0.012179045946260292, 'C2': 0, 'C3': 2.7432588728174234e-05, 'p0': 0.04553801347643801, 'p2': 0.0018022443124799303,
                    'q0': 0, 'q1': 0.02609037187916979, 'b1': 7.1780386096154185, 'b2': 0.053537268955100234, 'b3': 0.27965662935753677, 'z0': 1.4940081773441736, 'z1': 1.2718495543500672, 'ver': '2.0', 'mass': 1717, 'fuel_type': 'gasoline'}
# MIDSIZE_SUV_COEFFS = {'p1': 0.08239969771864666, 'C0': 0.3299962052097726, 'C1': 0.012441421108799908, 'C2': 0, 'C3': 4.343194215493537e-05, 'p0': 0.011909421841893171, 'p2': 0.0015970177813645265, 'q0': 0, 'q1': 0.020027881975315596, 'beta0': 0, 'b1': 1161.6582446141038, 'b2': 3.499999999999964, 'b3': 6.974946524693788, 'b4': -0.04802483189728902, 'b5': 3.195308567542204, 'ver': '1.0', 'mass': 1897, 'conversion': 33430.0, 'fuel_type': 'gasoline'}
# LIGHT_DUTY_PICKUP_COEFFS = {'p1': 0.11093857749806323, 'C0': 0.3907136679788762, 'C1': 0.011727132332537502, 'C2': 0, 'C3': 6.429806478809436e-05, 'p0': 0.007103876336589405, 'p2': 0.0016922164972516507, 'q0': 0, 'q1': 0.015723422004713525, 'beta0': 0, 'b1': 384.74170405872627, 'b2': 3.4999999920927154, 'b3': 4.825193120304821, 'b4': -0.04484165775966457, 'b5': 2.9596930151578404, 'ver': '1.0', 'mass': 2173, 'conversion': 33430.0, 'fuel_type': 'gasoline'}
# CLASS3_PND_TRUCK_COEFFS = {'p1': 0.31892608470405964, 'C0': 0.41411228845217835, 'C1': 0.0032954648215967977, 'C2': 0.0021935206123276513, 'C3': 6.495755513531579e-05, 'p0': 0.10475850621425438, 'p2': 0.0007843191385363194, 'q0': 0, 'q1': 0.019932155093294227, 'beta0': 0.19105753783308377, 'b1': 9999.999999832015, 'b2': 3.0286340128346825, 'b3': 19.770329080656857, 'b4': -0.06085166612240557, 'b5': 1.8976805391654787, 'ver': '1.0', 'mass': 5943, 'conversion': 33430.0, 'fuel_type': 'diesel'}
# CLASS8_TRACTOR_TRAILER_COEFFS = {'p1': 1.4773378205073733, 'C0': 0.39696714196228355, 'C1': 0.11251189426194752, 'C2': 0, 'C3': 0.00024898411590544496, 'p0': 0.015354525914402433, 'p2': 0.0009498671895212034, 'q0': 0, 'q1': 0.009898328089425493, 'beta0': 0.244388459468995, 'b1': 9999.99985610316, 'b2': 2.1296996469040512, 'b3': 36.875710683509865, 'b4': -0.034068260220609284, 'b5': 1.1859691015970248, 'ver': '1.0', 'mass': 25104, 'conversion': 33430.0, 'fuel_type': 'diesel'}


class PolyFitModel(object):
    """Simplified Polynomial Fit base energy model class.

    Calculate fuel consumption of a vehicle based on polynomial
    fit of Autonomie models. Polynomial functional form is
    informed by physics derivation and resembles power demand
    models.
    """

    def __init__(self, coeffs_dict):
        """
        Initialize the PolyFitModel class, using a dictionary of coefficients.

        It is not recommended to supply custom coefficients (it is also not possible to instantiate the PolyFitModel
        class directly since it is an abstract class). The child classes below instantiate versions of this class
        using the fitted coefficients provided in the *.mat files. For more details, see docs/PolyFitModels.pdf.

        Parameters
        ----------
        coeffs_dict : dict
            model coefficients, including:
            * "mass" (int | float): mass of the vehicle, for reference only
            * "C0" (float): C0 fitted parameter
            * "C1" (float): C1 fitted parameter
            * "C2" (float): C2 fitted parameter
            * "C3" (float): C3 fitted parameter
            * "p0" (float): p0 fitted parameter
            * "p1" (float): p1 fitted parameter
            * "p2" (float): p2 fitted parameter
            * "q0" (float): q0 fitted parameter
            * "q1" (float): q1 fitted parameter
            * "beta0" (float): minimum fuel consumption
            * "b1" (float): coeff 1 for modelling infeasible range
            * "b2" (float): coeff 2 for modelling infeasible range
            * "b3" (float): coeff 3 for modelling infeasible range
            * "z0" (float): z0 parameter for grade dependency
            * "z1" (float): z1 parameter for grade dependency
            * "conversion" (float): unit conversion from gal/hr to Watts
            * "v_max_fit" (float): assumed max velocity for modelling infeasible range
        """
        super(PolyFitModel, self).__init__()

        self.mass = coeffs_dict['mass']
        self.state_coeffs = np.array([coeffs_dict['C0'],
                                      coeffs_dict['C1'],
                                      coeffs_dict['C2'],
                                      coeffs_dict['C3'],
                                      coeffs_dict['p0'],
                                      coeffs_dict['p1'],
                                      coeffs_dict['p2'],
                                      coeffs_dict['q0'],
                                      coeffs_dict['q1'],
                                      coeffs_dict['z0'],
                                      coeffs_dict['z1']])
        self.beta0 = coeffs_dict['beta0']
        self.vc = coeffs_dict['vc']
        self.b1 = coeffs_dict['b1']
        self.b2 = coeffs_dict['b2']
        self.b3 = coeffs_dict['b3']
        self.fuel_type = coeffs_dict['fuel_type']

    def get_instantaneous_fuel_consumption(self, accel, speed, grade):
        """Calculate the instantaneous fuel consumption.

        Parameters
        ----------
        accel : float
            Instantaneous acceleration of the vehicle
        speed : float
            Instantaneous speed of the vehicle
        grade : float
            Instantaneous road grade of the vehicle

        Returns
        -------
        float
        """
        state_variables = np.array([1,
                                    speed,
                                    speed**2,
                                    speed**3,
                                    accel,
                                    accel * speed,
                                    accel * speed**2,
                                    max(accel, 0)**2,
                                    max(accel, 0)**2 * speed,
                                    grade,
                                    grade * speed])
        fc = np.dot(self.state_coeffs, state_variables)
        lower_bound = (speed <= self.vc) * self.beta0
        fc = max(fc, lower_bound)  # assign min fc when polynomial is below the min
        return fc * GRAMS_PER_SEC_TO_GALS_PER_HOUR[self.fuel_type]

    def flag_infeasible_accel(self, accel, speed, grade):
        """Return True if speed/accel pair outside of feasible range."""
        max_accel = self.b1 * math.exp(-self.b2 * speed) + self.b3
        return accel > max_accel.real

    def get_instantaneous_power(self, accel, speed, grade):
        """See parent class."""
        conversion = GRAMS_TO_JOULES[self.fuel_type] / GRAMS_PER_SEC_TO_GALS_PER_HOUR[self.fuel_type]
        return self.get_instantaneous_fuel_consumption(accel, speed, grade) * conversion


# class PFMCompactSedan(PolyFitModel):
#     """Simplified Polynomial Fit Model for CompactSedan.

#     Model is fitted to semi-principled model derived from Autonomie software.
#     """

#     def __init__(self, coeffs_dict=COMPACT_SEDAN_COEFFS):
#         super(PFMCompactSedan, self).__init__(coeffs_dict=coeffs_dict)


# class PFMMidsizeSedan(PolyFitModel):
#     """Simplified Polynomial Fit Model for MidsizeSedan.

#     Model is fitted to semi-principled model derived from Autonomie software.
#     """

#     def __init__(self, coeffs_dict=MIDSIZE_SEDAN_COEFFS):
#         super(PFMMidsizeSedan, self).__init__(coeffs_dict=coeffs_dict)


class PFM2019RAV4(PolyFitModel):
    """Simplified Polynomial Fit Model for 2019RAV4.

    Model is fitted to semi-principled model derived from Autonomie software.
    """

    def __init__(self, coeffs_dict=RAV4_2019_COEFFS):
        super(PFM2019RAV4, self).__init__(coeffs_dict=coeffs_dict)


# class PFMMidsizeSUV(PolyFitModel):
#     """Simplified Polynomial Fit Model for MidsizeSUV.

#     Model is fitted to semi-principled model derived from Autonomie software.
#     """

#     def __init__(self, coeffs_dict=MIDSIZE_SUV_COEFFS):
#         super(PFMMidsizeSUV, self).__init__(coeffs_dict=coeffs_dict)


# class PFMLightDutyPickup(PolyFitModel):
#     """Simplified Polynomial Fit Model for LightDutyPickup.

#     Model is fitted to semi-principled model derived from Autonomie software.
#     """

#     def __init__(self, coeffs_dict=LIGHT_DUTY_PICKUP_COEFFS):
#         super(PFMLightDutyPickup, self).__init__(coeffs_dict=coeffs_dict)


# class PFMClass3PNDTruck(PolyFitModel):
#     """Simplified Polynomial Fit Model for Class3PNDTruck.

#     Model is fitted to semi-principled model derived from Autonomie software.
#     """

#     def __init__(self, coeffs_dict=CLASS3_PND_TRUCK_COEFFS):
#         super(PFMClass3PNDTruck, self).__init__(coeffs_dict=coeffs_dict)


# class PFMClass8TractorTrailer(PolyFitModel):
#     """Simplified Polynomial Fit Model for Class8TractorTrailer.

#     Model is fitted to semi-principled model derived from Autonomie software.
#     """

#     def __init__(self, coeffs_dict=CLASS8_TRACTOR_TRAILER_COEFFS):
#         super(PFMClass8TractorTrailer, self).__init__(coeffs_dict=coeffs_dict)
