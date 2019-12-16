r"""
Riemann problem solvers for the Aw-Rascle-Zhang model
"""

from __future__ import absolute_import
import numpy as np
from six.moves import range

num_eqn = 2
num_waves = 2

from utils import pressure, pressure_1st_derivative, pressure_2nd_derivative, pressure_inverse


#	exact Riemann solver
def riemann(q_l, q_r, aux_l, aux_r, problem_data):
	# array shape
	num_rp = q_l.shape[1]

	# outputs
	wave = np.empty( (num_eqn, num_waves, num_rp) )
	s = np.zeros( (num_waves, num_rp) )
	amdq = np.zeros( (num_eqn, num_rp) )
	apdq = np.zeros( (num_eqn, num_rp) )

	# compute necessary quantities
	rho_l = q_l[0]
	rho_r = q_r[0]
	y_l = q_l[1]
	y_r = q_r[1]
	h_l = pressure(rho_l)
	h_r = pressure(rho_r)
	hp_l = pressure_1st_derivative(rho_l)
	hp_r = pressure_1st_derivative(rho_r)
	u_l = y_l / rho_l - h_l
	u_r = y_r / rho_r - h_r
	lam1_l = u_l - rho_l * hp_l
	lam1_r = u_r - rho_r * hp_r

	# compute intermediate states
	u_m = u_r
	rho_m = pressure_inverse(u_l+h_l-u_r)
	y_m = rho_m * (u_m + pressure(rho_m))
	hp_m = pressure_1st_derivative(rho_m)
	lam1_m = u_m - rho_m * hp_m

	# solve Riemann problems
	for i in range(num_rp):
		# 2-wave (right-going contact discontinuity)
		s[1, i] = u_m[i]
		wave[0, 1, i] = rho_r[i] - rho_m[i]
		wave[1, 1, i] = y_r[i] - y_m[i]
		apdq[:, i] += s[1, i] * wave[:, 1, i]

		# 1-wave
		if rho_l[i] < rho_m[i]:
			# shock wave
			s[0, i] = (rho_m[i] * u_m[i] - rho_l[i] * u_l[i]) / (rho_m[i] - rho_l[i])
			wave[0, 0, i] = rho_m[i] - rho_l[i]
			wave[1, 0, i] = y_m[i] - y_l[i]
			amdq[:, i] += min(s[0, i], 0) * wave[:, 0, i]
			apdq[:, i] += max(s[0, i], 0) * wave[:, 0, i]
		else:
			# rarefaction wave

			# artificially defined wave and speed for high-resolution methods
			s[0, i] = 0.5 * (lam1_l[i] + lam1_m[i])
			wave[0, 0, i] = rho_m[i] - rho_l[i]
			wave[1, 0, i] = y_m[i] - y_l[i]

			if lam1_m[i] < 0:
				# left-going rarefaction wave	
				amdq[0, i] += rho_m[i] * u_m[i] - rho_l[i] * u_l[i]
				amdq[1, i] += y_m[i] * u_m[i] - y_l[i] * u_l[i]
			elif lam1_l[i] > 0:
				# right-going rarefaction wave
				apdq[0, i] += rho_m[i] * u_m[i] - rho_l[i] * u_l[i]
				apdq[1, i] += y_m[i] * u_m[i] - y_l[i] * u_l[i]
			else:
				# centered rarefaction wave
				from scipy.integrate import ode
				def f(t, q):
				    return q / (-2 * q[0] * pressure_1st_derivative(q[0])\
				    	-q[0]**2 * pressure_2nd_derivative(q[0]))
				r = ode(f)
				r.set_initial_value(q_l[:, i], lam1_l[i])
				q_0 = r.integrate(0)
				fq_0 = [q_0[1]-q_0[0]*pressure(q_0[0]), q_0[1]**2/q_0[0]-q_0[1]*pressure(q_0[0])]
				amdq[0, i] += fq_0[0] - rho_l[i] * u_l[i]
				amdq[1, i] += fq_0[1] - y_l[i] * u_l[i]
				apdq[0, i] += rho_m[i] * u_m[i] - fq_0[0] 
				apdq[1, i] += y_m[i] * u_m[i] - fq_0[1]

	return wave, s, amdq, apdq


#	HLL solver
def hll(q_l, q_r, aux_l, aux_r, problem_data):
	# array shape
	num_rp = q_l.shape[1]

	# outputs
	wave = np.empty( (num_eqn, num_waves, num_rp) )
	s = np.zeros( (num_waves, num_rp) )
	amdq = np.zeros( (num_eqn, num_rp) )
	apdq = np.zeros( (num_eqn, num_rp) )

	# compute necessary quantities
	rho_l = q_l[0]
	rho_r = q_r[0]
	y_l = q_l[1]
	y_r = q_r[1]
	h_l = pressure(rho_l)
	h_r = pressure(rho_r)
	hp_l = pressure_1st_derivative(rho_l)
	hp_r = pressure_1st_derivative(rho_r)
	u_l = q_l[1] / rho_l - h_l
	u_r = q_r[1] / rho_r - h_r
	lam1_l = u_l - rho_l * hp_l
	lam1_r = u_r - rho_r * hp_r

	# HLL characteristic speeds
	s[0, :] = np.minimum(lam1_l, lam1_r)
	s[1, :] = np.maximum(u_l, u_r)

	# intermediate states
	rho_m = (rho_r * u_r - rho_l * u_l - s[1] * rho_r + s[0] * rho_l) / (s[0] - s[1])
	y_m = (y_r * u_r - y_l * u_l - s[1] * y_r + s[0] * y_l) / (s[0] - s[1])

	# compute waves
	wave[0, 0, :] = rho_m - rho_l
	wave[1, 0, :] = y_m - y_l
	wave[0, 1, :] = rho_r - rho_m
	wave[1, 1, :] = y_r - y_m

	# compute fluctuations
	for p in range(num_waves):
		amdq += np.minimum(s[p], np.zeros(num_rp)) * wave[:, p, :]
		apdq += np.maximum(s[p], np.zeros(num_rp)) * wave[:, p, :]

	return wave, s, amdq, apdq