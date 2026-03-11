import logging
import numpy as np
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import healpy as hp
import mhealpy as mp
from scoords.spacecraft_frame import SpacecraftFrame
from astropy import units as u
from astropy.coordinates import SkyCoord
from cosipy.response.ideal_response import (
    IdealComptonIRF,
    UnpolarizedIdealComptonIRF,
    RandomEventDataFromLineInSCFrame,
)
from cosipy.response.photon_types import (
    PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionInterface as PolDirESCPhoton, 
    PhotonWithDirectionAndEnergyInSCFrameInterface as DirESCPhoton,
    PhotonWithDirectionAndEnergyInSCFrame, PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConvention
)
from cosipy.polarization import StereographicConvention

# ===============================
# Configuration for events
# ===============================
energy = 1050 * u.keV

source_dir = SkyCoord(lon=0., lat=75., unit="deg", frame=SpacecraftFrame())
source_flux = 1. / (u.cm * u.cm * u.s)
duration = 1. * u.s
source_pd = 0.2
source_pa = 80. * u.deg
pol_convention = StereographicConvention()

#================================
# Healpix Map
#================================

order = 6
nside = 2 ** order
npix = hp.nside2npix(nside)

lon = source_dir.lon.to_value(u.rad)
lat = source_dir.lat.to_value(u.rad)

theta = 0.5 * np.pi - lat
phi = lon

vec = hp.ang2vec(theta, phi)

radius_deg = 30
radius_rad = np.deg2rad(radius_deg)

# Query pixels inside disc
pix_array = mp.query_disc(nside, vec, radius_rad)
#make pix_array all pixels within the healpy map
#pix_array = np.arange(npix)

print("There are", len(pix_array), "pixels in the disc.")
print(pix_array)

# Blank map
hpx_map = np.full(npix, hp.UNSEEN)

# ===============================
# IRFs
# ===============================

irf_pol = IdealComptonIRF.cosi_like()
irf_unpol = UnpolarizedIdealComptonIRF.cosi_like()

# ===============================
#Make exact amount of events:
def simulate_events():
    events = []

    d = RandomEventDataFromLineInSCFrame(
        irf=irf_unpol,
        flux=source_flux,
        duration=duration,
        energy=energy,
        direction=source_dir,
        polarized_irf=irf_pol,
        polarization_degree=source_pd,
        polarization_angle=source_pa,
        polarization_convention=pol_convention,
    )

    events.extend(d) 
    return events

events = simulate_events()

# Create i by j matrix to store probabilities
logging.info("Calculating probability matrix...")
prob_matrix = np.zeros((len(pix_array), len(events)))
aeff= []
for i in range(len(pix_array)):
    theta_i, phi_i = hp.pix2ang(nside, pix_array[i])
    photon = PhotonWithDirectionAndEnergyInSCFrame(phi_i,
                                               (0.5*np.pi) - theta_i,
                                               energy.to_value(u.keV))
    aeff.append(irf_unpol.effective_area_cm2([photon])[0])
    for j in range(len(events)):
        prob = irf_unpol.event_probability([(photon, events[j])])
        prob_matrix[i, j] = list(prob)[0]

def poisson_binned_log_likelihood(observed, expected):
    expected_safe = np.where(expected <= 0, 1e-10, expected)
    return np.sum(observed * np.log(expected_safe) - expected_safe)

aeff = np.array(aeff)

def unbinned_richardson_lucy(response, b_i,b_j, model_init, n_iter=20):
    """
    Perform Richardson-Lucy deconvolution (Unbinned).

    Parameters:
        response (ndarray): Response matrix, shape (n_data, n_model)
        model_init (ndarray): Initial model vector, shape (n_model,)
        n_iter (int): Number of iterations

    Returns:
        model (ndarray): Deconvolved model
        log_likelihoods (list): Log-likelihood at each iteration
    """
    model = model_init.copy()
    
    log_likelihoods = []

    for _ in range(n_iter):
        expectation = np.dot(response.T, model) + b_i
        log_likelihoods.append(poisson_binned_log_likelihood(1, expectation))

        coeff = np.einsum('ij,j->i',  response,1/expectation)

        R_j = np.asarray(aeff).flatten() * duration.to(u.s).value
        norm_coeff = np.zeros_like(coeff)
    
        np.divide(coeff, R_j , out=norm_coeff,
                  where=(coeff != 0) | (R_j != 0))

        model *= norm_coeff

    return model, log_likelihoods

events = np.array(events)
model = np.ones(prob_matrix.shape[0])  # Initial model guess

 # Backgrounds are 0 for now
b_i, b_j = np.zeros(prob_matrix.shape[1]), np.zeros(prob_matrix.shape[0])

log_like = []
logging.info("Starting Richardson-Lucy deconvolution...")

# ----- Plotting -----

# Plot each energy bin
for i in range(25):

    model[:], log_like = unbinned_richardson_lucy(prob_matrix,b_i,b_j, model, n_iter=i)
    # Create a full HEALPix map for plotting
    hpx_plot = np.zeros(npix)
    hpx_plot[pix_array] = model
    
    hp.mollview(hpx_plot, title=f"Iteration {i}", unit="arb", cmap="viridis")

    hp.projplot(theta, phi,
            marker='o',
            color='red',
            markersize=1)

    plt.tight_layout()
    plt.savefig(f"Iterations/iteration_{i:02d}.png")


