Attempt at replicating the observational analysis of Bronselaer et al 2020 (nature geoscience). In particular, exploring the effect of not removing GLODAP era shelf measurements prior to gridding and differencing. Grid boxes span both shelf and non shelf regions yet only the older time peiod includes shelf measurements, this biases anomalies between the older and more recent periods.

There is a logical order to these notebooks. 

1. natgeo_S2.ipynb explains and conducts all the interpolation and binning for SOCCOM and GLODAP raw profile data. Then produces maps according to Ben's supplementary figure 2. Analysis for GLODAP is conducted twise, with and without profiles collected in regions with bathymetry shallower than 2000m. Much of the nuts and bolts of the analysis is contained in SOCCOM_GLODAP_processing.py.

2. natgeo_Fig1_Fig2.ipynb uses the output files of above to remake the main figure papers (zonal mean view), again with and without bathymetry making.

3. natgeo_S7.ipynb replicate supplemetary figure 7 (Ben request)

4. Manual_QC-check.ipynb was trying to follow up ona. suggestion by Ben that the reason I wasn't able to replicate his low latitude results was to do with some profiles he reved manually due to QC, I couldn't find these profiles.

Hope this is helpful.
