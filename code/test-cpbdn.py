from __future__ import print_function
from builtins import input
from builtins import range

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco import util
from sporco import plot
plot.config_notebook_plotting()
import sporco.linalg as spl
import sporco.metric as sm
from sporco.admm import cbpdn
from sporco.admm import cbpdntv
import matplotlib.image as mpimg
def pad(x, n=8):

    if x.ndim == 2:
        return np.pad(x, n, mode='symmetric')
    else:
        return np.pad(x, ((n, n), (n, n), (0, 0)), mode='symmetric')


def crop(x, n=8):

    return x[n:-n, n:-n]
# img = util.ExampleImages().image('monarch.png', zoom=0.5, scaled=True,
#                                   gray=True, idxexp=np.s_[:, 160:672])


img = mpimg.imread('barbara1.png')
np.random.seed(12345)
imgn = img + np.random.normal(0.0, 0.1, img.shape)

print("Noisy image PSNR:    %5.2f dB" % sm.psnr(img, imgn))
npd = 16
fltlmbd = 5.0
imgnl, imgnh = util.tikhonov_filter(imgn, fltlmbd, npd)
D = util.convdicts()['G:8x8x32']
D = D[:,:,0:14]
# D = np.random.randn(8, 8, 14)
imgnpl, imgnph = util.tikhonov_filter(pad(imgn), fltlmbd, npd)
W = spl.irfftn(np.conj(spl.rfftn(D, imgnph.shape, (0, 1))) *
               spl.rfftn(imgnph[..., np.newaxis], None, (0, 1)),
               imgnph.shape, (0,1))
W = W**2
W = 1.0/(np.maximum(np.abs(W), 1e-8))

lmbda = 1.5e-2
mu = 0.005
opt1 = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 250,
              'HighMemSolve': True, 'RelStopTol': 3e-3, 'AuxVarObj': True,
              'L1Weight': W, 'AutoRho': {'Enabled': False}, 'rho': 4e2*lmbda})
opt2 = cbpdntv.ConvBPDNScalarTV.Options({'Verbose': True, 'MaxMainIter': 250,
             'HighMemSolve': True, 'RelStopTol': 3e-3, 'AuxVarObj': True,
             'L1Weight': W, 'AutoRho': {'Enabled': False}, 'rho': 4e2*lmbda})
opt3 = cbpdntv.ConvBPDNVectorTV.Options({'Verbose': True, 'MaxMainIter': 250,
             'HighMemSolve': True, 'RelStopTol': 3e-3, 'AuxVarObj': True,
             'L1Weight': W, 'AutoRho': {'Enabled': False}, 'rho': 4e2*lmbda})
a = cbpdn.ConvBPDN(D, pad(imgnh), lmbda, opt1, dimK=0)
b = cbpdntv.ConvBPDNScalarTV(D, pad(imgnh), lmbda, mu, opt2)
c = cbpdntv.ConvBPDNVectorTV(D, pad(imgnh), lmbda, mu, opt3)
X1 = a.solve()
X2 = b.solve()
X3 = c.solve()
imgdp1 = a.reconstruct().squeeze()
imgd1 = np.clip(crop(imgdp1) + imgnl, 0, 1)

imgdp2 = b.reconstruct().squeeze()
imgd2 = np.clip(crop(imgdp2) + imgnl, 0, 1)

imgdp3 = c.reconstruct().squeeze()
imgd3 = np.clip(crop(imgdp3) + imgnl, 0, 1)
print("ConvBPDN solve time: %5.2f s" % b.timer.elapsed('solve'))
print("Noisy image PSNR:    %5.2f dB" % sm.psnr(img, imgn))
print("bpdn PSNR: %5.2f dB" % sm.psnr(img, imgd1))
print("scalar image PSNR: %5.2f dB" % sm.psnr(img, imgd2))
print("vector image PSNR: %5.2f dB" % sm.psnr(img, imgd3))
fig = plot.figure(figsize=(21, 21))
plot.subplot(2, 3, 1)
plot.imview(img, title='Reference', fig=fig)
plot.subplot(2, 3, 2)
plot.imview(imgn, title='Noisy', fig=fig)
plot.subplot(2, 3, 3)
plot.imview(imgd1, title='Conv bpdn', fig=fig)
plot.subplot(2, 3, 4)
plot.imview(imgd2, title='scalar TV', fig=fig)
plot.subplot(2, 3, 5)
plot.imview(imgd3, title='vector TV', fig=fig)
fig.show()