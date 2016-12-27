import numpy as np

A = np.array([[0.5, 0.2, 0.3],[0.2, 0.4, 0.4],[0.4, 0.1, 0.5]])
phi = np.array([[0.8, 0.2],[0.1, 0.9],[0.5, 0.5]])
pi_0 = np.array([[0.5],[0.3],[0.2]])

result_list = []
s = 0
for i_0 in xrange(3):
  for i_1 in xrange(3):
    for i_2 in xrange(3):
      for i_3 in xrange(3):
        result = pi_0[i_0] * phi[i_0,0] * A[i_0,i_1] * phi[i_1,1] * \
              A[i_1,i_2] * phi[i_2,0] * A[i_2,i_3] * phi[i_3,1]
        s += float(result)
        result_list.append((result, (i_0, i_1, i_2, i_3)))

result_list = sorted(result_list, reverse=True)

for (p, (i0,i1,i2,i3)) in result_list[0:3]:
  print "sequence: (%d, %d, %d, %d)" % (i0, i1, i2, i3)
  # prior
  prior = pi_0[i0] * A[i0,i1] * A[i1,i2] * A[i2,i3]
  print "prior: %f" % prior

  # likelihood
  likelihood = p / prior
  print "likelihood: %f" % likelihood

  # posterior
  posterior = p / s
  print "posterior: %f" % posterior
  print
  print