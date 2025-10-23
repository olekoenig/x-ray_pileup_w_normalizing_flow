"""
Author: Ole Koenig (ole.koenig@cfa.harvard.edu)
 
A script to write a slurm script which creates eROSITA slew
lightcurves for different fluxes for a piled-up source given a certain
spectrum and using SIXTE's sixtesim.

----------
Changelog:
----------

- 2020-11-XX: Add eSASS pipeline 
- 2020-11-26: Add option to simulate multiple spectra for chi2 minimization 
- 2020-12-02: All output files except for spectra on scratch disks, mathpha for averaging
- 2021-01-29: Complete re-write
- 2021-07-21: Confirm that rewritten scripts can reproduce old setup, add clobber, add pha2pi
- 2021-08-06: Add image creation with evtool
- 2021-08-17: Parse verbose parameter into SIXTEs chatter to control output, add expression file writing in mathpha
- 2021-09-03: Fix bug in test parameter of sixtesoft that set SIXTE_USE_PSEUDO_RNG=1 by default
- 2021-09-16: Add function to average the simulated images
- 2021-09-18: Add attitude and GTI file computation based on CORRATT of evtool, fix mathpha error
- 2021-11-04: Add option to set verbosity=-1 such that no logfile is written
- 2021-11-22: Add systematic error to averaged spectrum
- 2021-12-01: Add function to average lightcurves
- 2024-08-09: Remove gtitype parameter from srctool (was removed in eSASS PipeProc030),
              Add automatic creation of output folders (spec/, lc/) as specified in config.py
              Rename 020 --> 820 as per eSASS convention (lightleak cannot be simulated)
- 2025-01-22: Move simputfile call into new wrapper within sixtesoft,
              Add sixtesim wrapper, add makespec commands and impact list creation
- 2025-10-20: Add logegrid flag, set to True (important for powerlaw)

------
To Do:
------

* Write automatized master ARF and RMF routine: both files are always
  the same between all simulations. Simulate with srctool todo=ALL
  once in beginning to write master ARF/RMF. PHAFILE
  keyword should be deleted (may lead to misunderstandings).

-----
Bugs:
-----

* Potential bug in creation of scratch folders when launching parallelized jobs
  on the same machine (folder can be already created by a different job).
"""

from pandas import read_csv
import shutil
import os
import subprocess
import shlex
import sys
from astropy.io import fits
import numpy as np

from config import SIXTEConfig
import attitude
import sixtesoft

config = SIXTEConfig()

class Pileupsim:
    def __init__(self, flux, parfile,
                 background = "yes",
                 n_sim=1, shortname="0", write_impactlist=False, verbose=0, clobber="no"):
        """
        :param flux: Flux of the source in units of erg/cm^2/s
        :param parfile: Parameter file in ISIS notation
        :param n_sim: Number of simulations to be averaged over
        :param shortname: A (unique) file naming for mathpha
        :param write_impactlist: Set to True if non-piledup spectrum should be written
        :param verbose: Verbosity control (-1: write no logfile) [default: 0]
        :param clobber: Clobber [default: no]
        """
        
        self.flux = flux
        self.parfile = parfile
        self.background = background
        self.isisprep = config.ISISPREP if config.ISISPREP else "none"
        self.n_sim = n_sim
        self.shortname = shortname  # see help of averageSpectra()
        self.write_impactlist = write_impactlist
        self.verbose = verbose

        # Define the names of the output files
        self.specparams = self._get_parameters_from_parfile(parfile)
        self.basename = self._get_basename(flux, self.specparams)

        self.evtfile_suffix = "piledup.fits"
        
        self.finalSpectrumName = f"{config.SPECDIR}{self.basename}_{self.evtfile_suffix}"
        if clobber == "no":
            self.check_if_final_spectrum_existent()

        self.finalBackSpecName = self.finalSpectrumName.replace("SourceSpec", "BackgrSpec")

        self._make_scratch_dirs()

        # Change into /scratch directory on hard-drive of machine, otherwise
        # temporary files are stored in /home, which slows down the simulation.
        os.chdir(config.SCRATCHDIR)

        if self.verbose >= 0:
            self.logfile = f"{config.LOGDIR}{self.basename}.log"
            print(f"Logfile: {self.logfile}")
            self.chatter = self.verbose
        else:
            self.logfile = None
            self.chatter = 0

        self.logegrid = False
        self.elow = 0.01  # [keV]
        self.ehigh = 15  # [keV]
            
        self.simputname = f"{config.SIMPUTDIR}{self.basename}.simput"
        self.evtfilelist = ["{}{}_tel{}_{}".format(config.EVTDIR, self.basename,
                                tm, self.evtfile_suffix) for tm in config.ACTIVE_TMS]
        self.merged_evtfile = f"{config.EVTDIR}{self.basename}.fits"
        self.ero_caleventslist = ["{}ero_calevents_{}_tel{}.fits".format(config.ESASSDIR, self.basename, tm)
                                  for tm in config.ACTIVE_TMS]
        self.finalImgName = f"{config.IMGDIR}{self.basename}_820_SkyImage.img"
        self.finalLCName = f"{config.LCDIR}{self.basename}_820_LightCurve.fits"
        self.srcphalist = []
        self.backphalist = []
        self.imglist = []
        self.lclist = []
        self.areascal = 1.0  # AREASCAL should be the same in all spectra

        if self.write_impactlist:
            self.implist_suffix = "nonpiledup.fits"
            self.impfilelist = ["{}{}_tel{}_{}".format(config.EVTDIR, self.basename,
                                tm, self.implist_suffix) for tm in config.ACTIVE_TMS]
            self.impactListSpectrum = f"{config.SPECDIR}{self.basename}_{self.implist_suffix}"
            self.merged_impfile = f"{config.EVTDIR}{self.basename}_{self.implist_suffix}"

        if self.logfile and os.path.exists(self.logfile):
            os.remove(self.logfile)

        if self.n_sim > 1:
            self.pfiles_dir = config.PFILESDIR + self.basename
            self.used_tools = ["simputfile", "simputsrc", "simputspec",
                               "sixtesim",
                               "ero_vis", "erosim", "runsixt", "ero_calevents", "makelc", "makespec",
                               "ftmerge", "fthedit", "fcalc", "mathpha",
                               "radec2xy", "srctool", "evtool"]
            self._write_pfiles()

    @staticmethod
    def _get_parameters_from_parfile(parfile):
        """
        A parser to return a dictionary of the interesting.
        Currently, "interesting" is everything except for ".norm"
        """
        df = read_csv(parfile, skiprows = 2, sep = r"\s+",
                      names = ['idx', 'param', 'tie-to', 'freeze', 'value', 'min', 'max', 'unit'])
        ret_dict = dict()
        for index, row in df.iterrows():
            param = row['param'].split(".")[-1]
            if param != "norm":
                ret_dict[param] = row['value']
        return ret_dict

    @staticmethod
    def _get_basename(flux, spec_params):
        """
        A function to create a human-readable file name.

        :param flux: The flux in units of 10^-10 erg/cm^2/s
        :param spec_params: A dictionary containing the interesting parameters as {par: value}
        """
        ret_str = "{:.6f}Em10cgs_".format(flux/1e-10)
        ret_str += "_".join([f"{val:.6f}{key}" for key, val in spec_params.items()])
        return ret_str

    @staticmethod
    def _softlink_spectra_to_shortname(shortname, filelist):
        filelist_short = []
        for n, file in enumerate(filelist):
            file_short = f"{shortname}{n}"
            os.symlink(file, file_short)
            filelist_short.append(file_short)
        return filelist_short
    
    def _make_scratch_dirs(self):
        """
        Store the SIMPUT, event files, and single spectra on scratch disk.
        """
        os.makedirs(config.SCRATCHDIR, exist_ok=True)
        os.makedirs(config.SIMPUTDIR, exist_ok=True)
        if self.n_sim > 0:
            os.makedirs(config.PFILESDIR, exist_ok=True)
        os.makedirs(config.EVTDIR, exist_ok=True)
        os.makedirs(config.ESASSDIR, exist_ok=True)

    def _write_pfiles(self):
        """
        In order not to get any weird interference effects, copy the pfiles to a local directory
        """
        os.makedirs(self.pfiles_dir, exist_ok=True)
        for tool in self.used_tools:
            shutil.copyfile(f"{config.GLOBAL_PFILES_DIR}/{tool}.par", f"{self.pfiles_dir}/{tool}.par")
        os.environ["PFILES"] = f"{self.pfiles_dir};{config.HEASOFT_PFILES}"

    def _check_area_scal(self, fp):
        """
        Check whether the AREASCAL keyword is the same in all spectra.
        """
        with fits.open(fp) as hdul:
            if hdul[1].header['AREASCAL'] != self.areascal:
                exit("ERROR: AREASCAL is {} in {}, not {}".format(hdul[0].header['AREASCAL'], fp, self.areascal))

    def _add_keywords(self, fp):
        self.execute(f"fthedit {fp} keyword=ANCRFILE operation=add value={os.path.basename(config.MASTER_ARF)}")
        self.execute(f"fthedit {fp} keyword=RESPFILE operation=add value={os.path.basename(config.MASTER_RMF)}")
        # EXPOSURE is not written unless ARF/ALL is specified in srctool.
        # Not calculating the ARF speeds up the simulations significantly.
        # self.execute(f"fthedit {fp} keyword=EXPOSURE operation=add value={self.exposure}")

    def check_if_final_spectrum_existent(self):
        if os.path.isfile(self.finalSpectrumName):
            sys.exit(f"File {self.finalSpectrumName} is already calculated. Exiting.")
        elif self.verbose > 0:
            print(f"Calculating {self.finalSpectrumName}")

    def execute(self, cmd):
        if self.logfile:
            with open(self.logfile, 'a+') as fp:
                fp.write(f" *** (execute): '{cmd}'\n" + "=" * 80 + "\n\n")
                ret_val = subprocess.run(shlex.split(cmd), shell=False, stdout=fp, stderr=fp)
        else:
            ret_val = subprocess.run(shlex.split(cmd), shell=False)
        ret_val.check_returncode()
            
    def create_simput(self):
        """
        Uses SIMPUT's simputfile function to create the SIMPUT for the given flux and parfile.
        The two main parameters are flux and parfile which come from the command line.
        """
        sixtesoft.simputfile(self.simputname, src_name = self.basename, srcflux = self.flux,
                             isisprep = self.isisprep, isisfile = self.parfile,
                             elow = self.elow, eup = self.ehigh, logegrid = self.logegrid,
                             emin = config.EMIN, emax = config.EMAX,
                             ra = config.RA, dec = config.DEC,
                             mjdref = config.MJDREF, chatter = self.chatter,
                             logfile=self.logfile)
        if self.verbose > 0:
            print(f"Wrote simput {self.simputname}")

    def write_attitude_and_gti(self):
        attitude.write_attitude()
        
        with fits.open(config.ATTITUDE_FILE) as hdul:
            tstart = hdul['ATTITUDE'].data['time'].min()  #: [s]
            exposure = hdul['ATTITUDE'].data['time'].max() - tstart  #: [s]

        sixtesoft.ero_vis(attitude=config.ATTITUDE_FILE, gtifile=config.MASTER_GTI, simput=self.simputname,
                          tstart=tstart, exposure=exposure, dt=config.DT,
                          visibility_range=1.0, logfile=self.logfile)

    def run_erosim(self):
        """
        Wrapper around sixte's erosim: Run the actual SIXTE simulation with erosim. Background is on.
        DEPRECATED SINCE SIXTE 3.0!
        """
        print("erosim is deprecated since SIXTE 3.0. Please use the run_sixtesim wrapper instead.")
        xmls = ["{}/erosita_{}.xml".format(config.XMLDIR, tm) for tm in range(1, 8)]

        impactlist = self.implist_suffix if self.write_impactlist else None
        
        sixtesoft.erosim(*xmls, prefix=f"{config.EVTDIR}{self.basename}_",
                         evtfile=self.evtfile_suffix, implist=impactlist,
                         attitude=config.ATTITUDE_FILE,
                         # tstart=self.tstart, expos=self.exposure,
                         gtifile=config.MASTER_GTI,
                         simput=self.simputname, background="yes",
                         seed=-1, dt=config.DT, chatter=self.chatter, mjdref=config.MJDREF,
                         logfile=self.logfile, test=False)

        # Delete the event files from TMs that are not active
        inactive_tms = [tm for tm in ["1", "2", "3", "4", "5", "6", "7"] if tm not in config.ACTIVE_TMS]
        unused_evtfilelist = ["{}{}_ccd{}_{}".format(config.EVTDIR, self.basename,
                                                     tm, self.evtfile_suffix) for tm in inactive_tms]
        [os.remove(fp) for fp in unused_evtfilelist]
        if self.write_impactlist:
            unused_impfilelist = ["{}{}_tel{}_{}".format(config.EVTDIR, self.basename,
                                    tm, self.implist_suffix) for tm in inactive_tms]
            [os.remove(fp) for fp in unused_impfilelist]
                
    def run_sixtesim(self):
        """
        Wrapper around sixtesim: Run the actual SIXTE simulation.
        """
        
        xmls = ",".join(["{}/erosita_{}.xml".format(config.XMLDIR, tm) for tm in config.ACTIVE_TMS])

        sixtesoft.sixtesim(xmls, prefix=f"{config.EVTDIR}{self.basename}_",
                           evtfile=self.evtfile_suffix,
                           attitude=config.ATTITUDE_FILE,
                           # tstart=self.tstart, expos=self.exposure,
                           gtifile=config.MASTER_GTI,
                           simput=self.simputname, background=self.background,
                           seed=-1, dt=config.DT, chatter=self.chatter, mjdref=config.MJDREF,
                           logfile=self.logfile, test=False)

    def merge_eventfiles(self):
        """
        Merge the event files from erosim with ftmerge
        """
        evtfilestr = " ".join(self.evtfilelist)
        self.execute(f"ftmerge \"{evtfilestr}\" {self.merged_evtfile} clobber=yes")

        if self.write_impactlist:
            impfilestr = " ".join(self.impfilelist)
            self.execute(f"ftmerge \"{impfilestr}\" {self.merged_impfile} clobber=yes")

    def ero_calevents(self):
        """
        Add additional columns/keywords to event file to make it consistent with eSASS
        """
        for ii in range(len(config.ACTIVE_TMS)):
            sixtesoft.ero_calevents(evtfile=self.evtfilelist[ii], eroevtfile=self.ero_caleventslist[ii],
                                    ccdnr=config.ACTIVE_TMS[ii], attitude=config.ATTITUDE_FILE,
                                    usepha=0, seed=-1, chatter=self.chatter,
                                    ra=config.RA, dec=config.DEC,
                                    projection="SIN", logfile=self.logfile,
                                    test=False)

    def esass_radec2xy(self):
        """
        Calculates X and Y sky pixel coordinates via tangential parallel projection.
        """
        for evt in self.ero_caleventslist:
            self.execute(f"radec2xy {evt} {config.RA} {config.DEC}")

    def sixte_radec2xy(self):
        self.execute(f"radec2xy {self.merged_evtfile} projection=TAN RefRA={config.RA} RefDec={config.DEC}")

    def run_evtool(self, n=1, events="no", image="yes"):
        eventfiles = " ".join(self.ero_caleventslist)
        outfile = f"{config.ESASSDIR}{self.basename}_820_SkyImage_{n}.fits"
        size = f"\"{config.IMGSIZE} {config.IMGSIZE}\""
        self.imglist.append(outfile)
        cmd = f"""evtool eventfiles=\"{eventfiles}\" outfile={outfile} \
events={events} image={image} size={size} center_position=\"auto\" clobber=yes"""
        self.execute(cmd)
        if self.verbose > 1:
            print(f" *** (run_evtool): output image is {outfile}")

    def get_exposure_from_ARF(self):
        with fits.open(config.MASTER_ARF) as hdul:
            self.exposure = hdul['SPECRESP'].header['EXPOSURE']
        if self.verbose > 0:
            print(f" *** (get_exposure_from_ARF): The exposure from the ARF is {self.exposure} s")

    def run_srctool(self, n=1, todo="ALL"):
        """
        Run eSASS srctool on the event files from ero_calevents to produce output products
        (spectrum, lightcurve).

        :param n: Index of the spectrum for naming, used to later average the spectra with mathpha
        :param todo: Specify which products should be calculated
        """

        suffix = f"_{n}.fits"
        self.n_ebands = 3  # Number of energy bands
        
        eventfiles = " ".join(self.ero_caleventslist)
        
        [self.execute("fthedit {} keyword=GTI_SEL operation=add value=GTI".format(fp)) for fp in self.ero_caleventslist]

        cmd = f"""srctool \
eventfiles=\"{eventfiles}\" \
srccoord=\"fk5;{config.RA},{config.DEC}\" \
prefix={config.ESASSDIR}{self.basename}_ \
suffix={suffix} \
todo='{todo}' \
insts=\"{' '.join(config.ACTIVE_TMS)}\" \
writeinsts=\"8\" \
srcreg=\"{config.SRCREG}\" \
backreg=\"{config.BKGREG}\" \
lctype=REGULAR+ lcpars={config.DT} \
lcemin="0.2 0.6 2.0" lcemax="0.6 2.0 8.0" \
lcgamma=2.0 tstep=1.0E-3 xgrid="1.0 1.0" \
psftype=2D_PSF \
flagsel=0 pat_sel=15 \
clobber=yes"""
        self.execute(cmd)

        srcfile = f"{config.ESASSDIR}{self.basename}_820_SourceSpec_00001{suffix}"
        backfile = srcfile.replace("SourceSpec", "BackgrSpec")

        self._check_area_scal(srcfile)
        self._check_area_scal(backfile)

        if not "ALL" in todo and not "ARF" in todo:
            self.get_exposure_from_ARF()
            self._add_keywords(srcfile)
            self._add_keywords(backfile)
        
        if self.verbose > 1:
            print(f" *** (run_srctool): output spectrum of run {n+1}/{self.n_sim}: {srcfile}")

        # Keep track of spectra and lightcurves for later averaging
        self.srcphalist.append(srcfile)
        self.backphalist.append(backfile)
        if "LC" in todo:
            self.lclist.append(srcfile.replace("SourceSpec", "LightCurve"))

        # Remove SRCTOOL products of TMs (120, 220,...) and leave only merged spectrum
        # srcfile_tms = [f"{config.ESASSDIR}{self.basename}_{tm}20_SourceSpec_00001{suffix}" for tm in config.ACTIVE_TMS]
        # [os.remove(fp) for fp in srcfile_tms]
        # bkgfile_tms = [f"{config.ESASSDIR}{self.basename}_{tm}20_BackgrSpec_00001{suffix}" for tm in config.ACTIVE_TMS]
        # [os.remove(fp) for fp in bkgfile_tms]

        if "ARF" in todo:
            arffiles = [f"{config.ESASSDIR}{self.basename}_{tm}20_ARF_00001{suffix}" for tm in config.ACTIVE_TMS]
            [os.remove(fp) for fp in arffiles]

    def run_makelc(self):
        """
        Produce a lightcurve with the SIXTE function "makelc" assuming
        TSTART and EXPOSURE from the attitude file.
        """
        with fits.open(config.ATTITUDE_FILE) as hdul:
            self.tstart = hdul['ATTITUDE'].header['TSTART']
            self.exposure = hdul['ATTITUDE'].header['TSTOP'] - self.tstart

        self.merge_eventfiles() 
        sixtesoft.makelc(evtfile=self.merged_evtfile,
                         lightcurve=self.merged_evtfile.replace(".fits", ".lc"),
                         tstart=self.tstart, length=self.exposure, dt=config.DT,
                         emin=0.21, emax=0.3, chanmin=-1, chanmax=-1,
                         gtifile=None, chatter=self.chatter, clobber="yes", history="true",
                         precmd="", logfile=self.logfile, test=False)

    def run_makespec(self):
        self.merge_eventfiles()
        self.sixte_radec2xy()

        sixtesoft.makespec(evtfile=self.merged_evtfile, spectrum=self.finalSpectrumName,
                           rsppath=config.XMLDIR, logfile=self.logfile)
        self.add_spec_param_keywords(self.finalSpectrumName)
        self._add_keywords(self.finalSpectrumName)
        
        if config.ANNULI_REGFILES:
            for regfile in config.ANNULI_REGFILES:
                pha_name = "{}_{}.fits".format(os.path.splitext(self.finalSpectrumName)[0],
                                               os.path.splitext(os.path.basename(regfile))[0])
                sixtesoft.makespec(evtfile=self.merged_evtfile, spectrum=pha_name,
                                   rsppath=config.XMLDIR, logfile=self.logfile, regfile=config.PILEUPSIM_DIR+regfile)
                self.add_spec_param_keywords(pha_name)
                self._add_keywords(pha_name)


    def add_spec_param_keywords(self, fp):
        for key, val in self.specparams.items():
            self.execute(f"fthedit {fp} keyword={key} operation=add value={val}")
        
        self.execute(f"fthedit {fp} keyword=SRC_FLUX operation=add value={self.flux}")
        self.execute(f"fthedit {fp} keyword=EMIN operation=add value={config.EMIN}")
        self.execute(f"fthedit {fp} keyword=EMAX operation=add value={config.EMAX}")
        self.execute(f"fthedit {fp} keyword=N_SIM operation=add value={self.n_sim}")
        self.execute(f"fthedit {fp} keyword=PARFILE operation=add value=\"\'{self.parfile}\'\" longstring=YES")

    def run_mathpha(self, infilelist, outfile):
        expr_file = f"{config.ESASSDIR}{self.shortname}_mathpha.txt"
        expr = f"({'+'.join(infilelist)})/{self.n_sim}"
        with open(expr_file, 'w') as fp:
            fp.write(expr)

        # Write expression into the logfile
        if self.verbose > 0:
            with open(self.logfile, 'a+') as fp:
                fp.write(f" *** (run_mathpha): Expression is '{expr}'\n\n")

        cmd = f"""mathpha \
expr=\"@{expr_file}\" \
units='COUNTS' \
outfil='{outfile}' \
exposure='CALC' \
areascal='%' \
ncomments=0 \
properr=YES \
clobber=yes"""
        self.execute(cmd)
        os.remove(expr_file)
        
        # Bug in mathpha as in help: "At the current time, unfortuantely one CANNOT specify the name of (say)
        # the response matrix to be written as the value of the RESPFILE keyword via the parameter rmfile"
        # --> write ANCRFILE, RESPFILE manually to file
        self._add_keywords(outfile)

    def average_spectra(self):
        """
        .. note:: Why do we need a "shortname"? The mathpha expr
          parameter becomes very long if many spectra are averaged.
          We need to softlink the names onto a unique short name, which
          is currently simply a running integer over all parallelized jobs.
          The notation is:
          - background: b{shortname}{n}
          - source: s{shortname}{n}
          where n goes from 0...n_sim
        """

        if not os.path.exists(config.SPECDIR):
            os.makedirs(config.SPECDIR)
        
        # Change to folder where spectra are stored such that mathpha call
        # doesn't require absolute paths
        os.chdir(config.ESASSDIR)

        if self.n_sim > 1:
            # Average source spectra
            src_outfile = f"s{self.shortname}_m.fits"
            self.run_mathpha(infilelist=self._softlink_spectra_to_shortname(f"s{self.shortname}", self.srcphalist),
                             outfile=src_outfile)
            [os.unlink(f"s{self.shortname}{n}") for n in range(self.n_sim)]

            # Average background spectra
            back_outfile = f"b{self.shortname}_m.fits"
            self.run_mathpha(infilelist=self._softlink_spectra_to_shortname(f"b{self.shortname}", self.backphalist),
                             outfile=back_outfile)
            [os.unlink(f"b{self.shortname}{n}") for n in range(self.n_sim)]

        else:
            src_outfile = self.srcphalist[0]
            back_outfile = self.backphalist[0]

        self._check_area_scal(src_outfile)
        self._check_area_scal(back_outfile)
            
        # Add background keywords
        self.execute(f"fthedit {src_outfile} keyword=BACKFILE operation=add longstring=YES value='{os.path.basename(self.finalBackSpecName)}'")
        self.execute(f"fthedit {back_outfile} keyword=BACKFILE operation=add value='NONE'")
        
        # Move file to global disk and change back into working dir.
        shutil.move(src_outfile, self.finalSpectrumName)
        shutil.move(back_outfile, self.finalBackSpecName)

        # The error in the averaged output spectrum is not computed
        # properly, because mathpha assumes Poisson error
        # (POISSERR=T), i.e. sqrt(COUNTS), where COUNTS=TOTCOUNTS/N
        # and TOTCOUNTS=sum_i counts_i.  Due to the averaging, the
        # statistical error must be sqrt(TOTCOUNTS)/N=sqrt(COUNTS/N)
        err = f"sqrt(COUNTS/(1.0*{self.n_sim})+({config.SYS_ERR}*COUNTS)^2)"  # Add statistical and systematic error in quadrature
        self.execute(f"fcalc {self.finalSpectrumName} {self.finalSpectrumName} STAT_ERR \"{err}\" clobber=yes")
        self.execute(f"fcalc {self.finalBackSpecName} {self.finalBackSpecName} STAT_ERR \"{err}\" clobber=yes")
        # Poisson error not correct (POISSERR=F) as a STAT_ERR column is present
        self.execute(f"fthedit {self.finalSpectrumName} keyword=\"POISSERR\" value=F operation=add")
        self.execute(f"fthedit {self.finalBackSpecName} keyword=\"POISSERR\" value=F operation=add")

        self.add_spec_param_keywords(self.finalSpectrumName)

        if self.verbose > 0:
            print(f" *** (average_spectra): Final spectrum is {self.finalSpectrumName}")

    def average_images(self):
        """
        Loads all simulated images and averages them.
        """

        if not os.path.exists(config.IMGDIR):
            os.makedirs(config.IMGDIR)

        with fits.open(self.imglist[0]) as fp_out:
            summed_img = fp_out[0].data
            for img in self.imglist[1:len(self.imglist)]:
                with fits.open(img) as hdul:
                    summed_img += hdul[0].data
            fp_out[0].data = summed_img / self.n_sim
            fp_out.writeto(self.finalImgName, overwrite=True)
        if self.verbose > 1:
            print(f" *** (average_images): {self.finalImgName}")

    def average_lightcurves(self):
        if not os.path.exists(config.LCDIR):
            os.makedirs(config.LCDIR)

        n_lcs = len(self.lclist)
        
        # Load the first lightcurve to preserve the FITS structure and then
        # start adding from LC #2
        with (fits.open(self.lclist[0]) as fp_out):
            lc_0 = fp_out["RATE"].data
            exposure_0 = max(lc_0["TIME"]) - min(lc_0["TIME"])

            lc_out = lc_0  # lc_out is the dataset that is averaged
            for lc in self.lclist[1:len(self.lclist)]:
                with fits.open(lc) as hdul:
                    lc_data = hdul["RATE"].data
                    exposure_lc = max(lc_data["TIME"]) - min(lc_data["TIME"])

                    # Some testing whether the lightcurves fit to each other
                    if not any(np.equal(lc_0["BACKRATIO"], lc_data["BACKRATIO"])):
                        print(" *** WARNING (average_lightcurves): The BACKRATIO of the lightcurves to be averaged is not the same!")
                    if exposure_0 != exposure_lc:
                        print(" *** WARNING (average_lightcurves): The exposures of the lightcurves to be averaged are not the same!")
                    if not any(np.equal(lc_0["TIMEDEL"], lc_data["TIMEDEL"])):
                        print(" *** WARNING (average_lightcurves): The TIMEDEL of the lightcurves to be averaged is not the same!")

                    for ii in range(self.n_ebands):
                        if not any(np.equal(lc_0["FRACEXP"][:,ii], lc_data["FRACEXP"][:,ii])):
                            print(" *** WARNING (average_lightcurves): The FRACEXP of the lightcurves to be averaged is not the same!")

                        # Sum up to get total counts
                        lc_out["COUNTS"][:,ii] += lc_data["COUNTS"][:,ii]
                        lc_out["BACK_COUNTS"][:,ii] += lc_data["BACK_COUNTS"][:,ii]

            # Calculate the averaged rate (see https://erosita.mpe.mpg.de/edr/DataAnalysis/srctool_doc.html -> Output files)
            for ii in range(self.n_ebands):
                lc_out["RATE"][:,ii] = (lc_out["COUNTS"][:,ii] - lc_out["BACK_COUNTS"][:,ii]*lc_out["BACKRATIO"]) \
                                       / (n_lcs * lc_out["FRACEXP"][:,ii] * lc_out["TIMEDEL"])
                # Error scales as sqrt(totcounts)/N = sqrt(counts)/sqrt(N)
                lc_out["RATE_ERR"][:,ii] = np.sqrt(lc_out["COUNTS"][:,ii] + lc_out["BACK_COUNTS"][:,ii]*lc_out["BACKRATIO"]) \
                                           / (n_lcs * lc_out["FRACEXP"][:,ii] * lc_out["TIMEDEL"])

                # Overwrite counts with mean counts
                # lc_out["COUNTS_ERR"][:,ii] = np.sqrt(lc_out["COUNTS"][:,ii]) / n_lcs;
                lc_out["COUNTS"][:,ii] = lc_out["COUNTS"][:,ii] / np.sqrt(n_lcs)

            fp_out.writeto(self.finalLCName, overwrite=True)

        self.add_spec_param_keywords(self.finalLCName)
        [os.remove(fp) for fp in self.lclist]
        
        if self.verbose > 1:
            print(f" *** (average_lightcurves): {self.finalLCName}")

    def clean_up(self):
        """
        Clean-up all temporary files written to the scratch disk (SCRATCHDIR).
        """
        if self.n_sim > 1:
            shutil.rmtree(self.pfiles_dir)
        os.remove(self.simputname)
        os.remove(self.merged_evtfile)
        [os.remove(fp) for fp in self.evtfilelist]
        #[os.remove(fp) for fp in self.ero_caleventslist]
        #[os.remove(fp) for fp in self.srcphalist]
        #[os.remove(fp) for fp in self.backphalist]
        # arflist = [fp.replace("SourceSpec", "ARF") for fp in self.srcphalist]
        # [os.remove(fp) for fp in arflist]


def main():
    pileupsim = Pileupsim(flux = 150, parfile = "test.par", n_sim = 2)
    print(pileupsim)

if __name__ == "__main__":
    main()
