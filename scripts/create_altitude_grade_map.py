import pandas as pd
import numpy as np
import pickle
import os
from scipy.interpolate import UnivariateSpline

grade_path = os.path.abspath(os.path.join(__file__, '../../dataset/Eastbound_grade_fit.csv'))
altitude_path = os.path.abspath(os.path.join(__file__, '../../dataset/Eastbound_elevation_fit.csv'))
grade_msg = pd.read_csv(grade_path)
altitude_msg = pd.read_csv(altitude_path)

mi_to_m = 1609.344

################################################################################################
# Make the spline for road grade

# Arbitrary number of points
num_points = 5000
# end in miles
end_mi = grade_msg['interval_end'][grade_msg.shape[0]-1]
end_m = end_mi * mi_to_m
x = np.linspace(0, end_m, num_points)
y = np.ones(num_points)
d = {'x': x, 'y': y}
df_grade = pd.DataFrame(data=d)

# Calculate xs and ys for grade
idx = len(grade_msg) - 1
for _, row in df_grade.iterrows():
    if (end_mi - np.around(row['x'] / mi_to_m, 8) < grade_msg['interval_start'][idx]):
        idx -= 1
    coeffs = np.array([grade_msg['slope'][idx], grade_msg['intercept'][idx]])
    x_vec = np.array([end_mi - (row['x'] / mi_to_m), 1])
    row['y'] = np.dot(coeffs, x_vec)

roadgrade_map = UnivariateSpline(df_grade['x'], df_grade['y'], k=1, s=0, ext=0)
road_grade = {'road_grade_map': roadgrade_map,
              'bounds': (0, end_m)}

with open(os.path.abspath(os.path.join(__file__, '../../dataset/road_grade_interp.pkl')), 'wb') as fp:
    pickle.dump(road_grade, fp)

################################################################################################
# Make the spline for altitude

# Arbitrary number of points
num_points = 5000
# end in miles
end_mi = altitude_msg['interval_end'][altitude_msg.shape[0]-1]
end_m = end_mi * mi_to_m
x = np.linspace(0, end_m, num_points)
y = np.ones(num_points)
d = {'x': x, 'y': y}
df_altitude = pd.DataFrame(data=d)

# Calculate xs and ys for altitude
idx = len(altitude_msg) - 1
for _, row in df_altitude.iterrows():
    if (end_mi - np.around(row['x'] / mi_to_m, 8) < grade_msg['interval_start'][idx]):
        idx -= 1
    coeffs = np.array([altitude_msg['quadratic_term'][idx], altitude_msg['linear_term'][idx], altitude_msg['intercept'][idx]])
    # Since the fit is reflected, we are actually querying from the other side and in terms of miles
    x_val = end_mi - (row['x'] / mi_to_m)
    x_vec = np.array([x_val**2, x_val, 1])
    row['y'] = np.dot(coeffs, x_vec)

altitude_map = UnivariateSpline(df_altitude['x'], df_altitude['y'], k=2, s=0, ext=0)
altitude = {'altitude_map': altitude_map,
            'bounds': (0, end_m)}

with open(os.path.abspath(os.path.join(__file__, '../../dataset/altitude_interp.pkl')), 'wb') as fp:
    pickle.dump(altitude, fp)
