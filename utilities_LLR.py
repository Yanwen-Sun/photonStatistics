import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def p0_dist(beta, kbar):
    #k = 0
    if np.isclose(beta,0): 
        _kbar = np.float_(kbar)
        p0 = np.exp(-_kbar)
    else: 
        p0 = (1/(1+kbar*beta)**(1./beta))
    return p0

def p_dist_rec(beta, kbar, k):
    #k>=1, recursive equation, equation (6)
    p = p0_dist(beta, kbar)
    for i in range(k):
        p *= kbar*(i*beta+1)/(kbar*beta+1)/(1+i)
    return p

## The multi-photon probability based on recursive equation
def p_dist(beta, kbar):
    p_calc = np.zeros(kbar.shape)
    p_calc[0] = p0_dist(beta, kbar[0])
    for k in range(1, kbar.shape[0]):
        p_calc[k] = p_dist_rec(beta, kbar[k], k)
    return p_calc

# calculation of chi^2, equation (4)
def chisqs(p,kavg,beta,Np, nphot):
    kavg = np.tile(kavg,(nphot+1,1))
    if (type(Np) is not int) or (type(Np) is not float):
        Np = np.tile(Np, (nphot+1,1))
    chi2 = -2*np.nansum((p*Np*np.log(1/p*p_dist(beta, kavg))))
    return chi2

#contrast calculation
def getContrast_beta(ps, Np, beta2, nphot = 2, beta1 = 0.001):
    nn = int(np.round((beta2-beta1)*1000+1))
    #calculate chi^2 with a step size of 0.001 for beta, considering up to nphot probability
    betas = np.linspace(beta1, beta2, nn)
    chi2 = np.zeros(betas.size)
    for ii,beta in enumerate(betas):
        chi2[ii] = chisqs(p = ps[:nphot+1], kavg = ps[-1],beta = beta, Np = Np, nphot = nphot)
    pos = np.argmin(chi2)
    beta0 = betas[pos]
    #curvature as error analysis, equation (5)
    dbeta = betas[1] - betas[0]
    ndbeta = 5
    delta_beta = np.sqrt(2*dbeta**2*ndbeta**2/(chi2[pos+ndbeta]+chi2[pos-ndbeta]-2*chi2[pos]))
    return [beta0, delta_beta], chi2, betas