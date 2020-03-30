# ANODE
Demonstration code for ANOmaly Detection with Density Estimation (https://arxiv.org/abs/2001.04990)

Code takes the LHC Olympics 2020 high-level feature sets as input. These can be downloaded here:

https://doi.org/10.5281/zenodo.3733786

This code uses a version of the conditional MAF that was adapted from here:

https://github.com/ikostrikov/pytorch-flows

=============================================================

Step 1:

Train the conditional MAF density estimators on the training data. This consists of the first half of the background events plus 500 signal events. The training data is divided up into the signal region (SR) [3.3<mJJ<3.7] and the sideband region (SB) [mJJ<3.3 or mJJ>3.7].  Density estimators are separately trained on each region.

python train_LHCORD_withsig.py --epochs=50 --minmass=3.3 --maxmass=3.7 --label=withsig --noshuffle --datashift=0. > & log_withsig &

Step 2:

Evaluate the trained density estimators on the testing data in the SR. This consists of the second half of the background events plus some number of signal events.

python eval_LHCORD_withsig.py --innermodel=INNERMODEL --outermodel=OUTERMODEL

This produces a number of npy files:

outermodel+'_logPfull_actual' -- the SR log density estimate evaluated on the second half of background plus ALL the remaining signal

outermodel+'_logPbg_actual' -- the SB log density estimate interpolated into the SR, evaluated on the second half of background plus ALL the remaining signal

outermodel+'_logPfull' -- the SR log density estimate evaluated on the second half of background plus 500 new signal events

outermodel+'_logPbg' -- the SB log density estimate interpolated into the SR, evaluated on the second half of background plus 500 new signal events


