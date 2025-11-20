# Modeling X-ray pile-up with a normalizing flow

Author: Ole Koenig, Harvard-Smithsonian Center for Astrophysics (ole.koenig@cfa.harvard.edu)

This simulation-based inference approach models the effect of _photon pile-up_ that occurs in silicon-based X-ray detectors. Loosely speaking, pile-up occurs if a celestial X-ray source is so bright that the detector cannot identify individual X-ray photons anymore. This highly non-linear effect causes distortions in the energy spectrum and can lead to data loss, making it challenging to analyze the data. We present a normalizing flow that takes piled-up observations as an input and outputs the posterior distributions of astrophysical parameters.

The study is published in Koenig et al., Machine Learning and the Physical Sciences Workshop, NeurIPS 2025: https://arxiv.org/abs/2511.11863

The training data is simulated using the SIXTE simulator (Dauser et al., A&A 630, 66, 2019). Successful application of simulations for understanding pattern and energy pile-up has been shown, among others, in Koenig et al., Nature 605, 248, 2022.
