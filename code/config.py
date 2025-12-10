import os
from dataclasses import dataclass

@dataclass(frozen=True)
class MLConfig:
    data_neural_network: str = "/pool/burg1/astroai/pileup/data_neural_network/"

    # Hyperparameters of neural network and related parameters
    dataloader_random_seed: int = 42
    dim_input_parameters: int = 1024  # channels in spectrum
    dim_output_parameters: int = 3
    batch_size: int = 512
    learning_rate: float = 0.001

@dataclass(frozen=True)
class ModelConfig:
    names = [r"$kT$ [keV]", r"Flux [$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}$]", r"$N_\mathrm{H}$ [$10^{22}\,\mathrm{cm}^{-2}$]"]

@dataclass(frozen=True)
class SIXTEConfig():
    """
    Configuration parameters for eROSITA pile-up simulations with SIXTE.
    """
    PILEUPSIM_DIR = os.getenv("HOME") + "/work/astroai/code/sixte_simulations/"

    ACTIVE_TMS = ["1", "2", "3", "4", "6"]  #: Numbers of active TMs in the real observation
    MJDREF = 51543.875  #: [days]
    DT = 0.5  #: Time resolution [s]

    RA = 77.283792  #257.283792-180 due to definition in SIXTE  # [decimal degrees]
    DEC = -37.511361  # [decimal degrees]
    SRCREG = os.getenv("HOME") + "/work/sources/gloria_novae/V1710Sco_src.reg"
    BKGREG = os.getenv("HOME") + "/work/sources/gloria_novae/V1710Sco_bkg.reg"

    ANNULI_REGFILES = ["circle0.reg", "annulus1.reg", "annulus2.reg", "annulus3.reg"]  # Need to fix to full path

    # Path to the measured eventfile as produced by eSASS' evtool (used to
    # create attitude file)
    ERO_EVENTFILE = "/userdata/data/koenig/novae4ole/V1710Sco_em04.evt"

    # Systematic uncertainty assigned to the spectrum due to pile-up modeling
    SYS_ERR = 0.1

    # Define energy range for SIMPUT creation (source flux range)
    EMIN = 0.2  #: [keV]
    EMAX = 2.0  #: [keV]

    # Parameters specific to the data paths and computational setup
    XMLDIR = os.getenv("HOME") + "/git/software/sixte_gits/instruments/srg/instruments/srg/erosita/"

    # Folders of temporary data products (somewhere with high-speed I/O!).
    # If a big grid is calculated some products should to be deleted after each run.
    # The grid simulation is sped up if one writes these files onto a local disk.
    SCRATCHDIR = os.getenv("HOME") + "/scratch/"
    SIMPUTDIR = SCRATCHDIR + "simput/"
    EVTDIR = SCRATCHDIR + "evt/"
    ESASSDIR = SCRATCHDIR + "esass/"

    # Handling of the pfiles is important only when multiple simulations
    # are done in parallel.
    PFILESDIR = SCRATCHDIR + "pfiles/"
    HEASOFT_PFILES = os.getenv("HEADAS") + "/syspfiles" if os.getenv("HEADAS") != None else ""
    GLOBAL_PFILES_DIR = os.getenv("HOME") + "/pfiles"

    # Folders of the data products on global disk (these products are not deleted)
    DATADIR = "/pool/burg1/astroai/pileup/sims/"
    LCDIR = DATADIR + "lc/"
    IMGDIR = DATADIR + "img/"
    LOGDIR = DATADIR + "log/"
    SPECDIR = DATADIR + "spec_piledup/"
    TARGET_SPECDIR = DATADIR + "spec_nonpiledup/"
    ISISPARFILEDIR = DATADIR + "pars/"

    # If we always simulate the same slew, we don't need to always compute the
    # GTI file (ero_vis) and write RMF/ARF (srctool) but can calculate it
    # once and use it as "master files"
    MASTER_ARF = PILEUPSIM_DIR + "srctool_master.arf"  # not needed if TODO=ALL in srctool
    MASTER_RMF = PILEUPSIM_DIR + "srctool_master.rmf"  # not needed if TODO=ALL in srctool
    ATTITUDE_FILE = PILEUPSIM_DIR + "master.att"  # created by write_attitude_and_gti
    MASTER_GTI = PILEUPSIM_DIR + "master.gti"  # write_attitude_and_gti
    ISISPREP = "none"

    # Image size of evtool output
    IMGSIZE = 70  #: pixel, default: 1024
