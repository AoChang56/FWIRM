# Full Waveform Inversion constrained with well log data
This package is developed based on the paper [Full waveform inversion using Random Mixing](https://doi.org/10.1016/j.cageo.2022.105041), which further includes the well log information as the direct observation for the inversion.

The inversion work is implemented by the geostatistical algorithm-**Random Mixing**-which requires two geostatistical characterizations, [marginal distribution](https://en.wikipedia.org/wiki/Marginal_distribution) and [spatial variogram](https://en.wikipedia.org/wiki/Variogram) , as primary information for the whole inversion processing.

Assume the limited velocity information is known via the well log in the targeted area.

<img src="https://github.com/user-attachments/assets/ce172d19-5980-4d8a-97e4-c18e74c7e1ba" alt="wells" width="400" height="100">

The experimental marginal distribution for the inversion can be estimated by this prior data.

<img src="https://github.com/user-attachments/assets/609c46e2-0051-4136-b67a-369945d26693" alt="kde gamma" width="300" height="200">


# Installation
git clone https://github.com/AoChang56/FWIRM.git

cd FWIRM

> [!IMPORTANT]
> Make sure all the required libraries have been installed as noted in the extra.txt before running. The parallel computing in the frequency domain needs to be run in the HPC.

# Implementation

run *forward_modelling.py* to get the "observation data" firstly. 

run *RMFWI.py* for an ensemble of velocity realisations.

> [!NOTE]
> The estimation of geostatistics characterisations in the inversion processing needs to be consistent with the input velocity data in the forward modelling.




