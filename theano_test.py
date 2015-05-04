import theano
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

rng = np.random.RandomState(12321)
theano_rng = RandomStreams(rng.randint(2 ** 30))
res = theano_rng.binomial(size=np.zeros((1,5)).shape,n=1,p=[0.1,0.1,0.1,0.1,0.1])

print res.eval()
