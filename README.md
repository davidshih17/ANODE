# ANODE
Demonstration code for ANOmaly Detection with Density Estimation (https://arxiv.org/abs/2001.04990)

Code takes the LHC Olympics 2020 high-level feature sets as input. These can be downloaded here:

https://doi.org/10.5281/zenodo.3733786

Step 1:

Train the conditional MAF density estimators on the training data (SR = 'inner' and SB = 'outer'). This consists of the first half of the background events plus 500 signal events.

python train_LHCORD_withsig.py --epochs=50 --minmass=3.3 --maxmass=3.7 --label=withsig --noshuffle --datashift=0. > & log_withsig &

Step 2:

Evaluate the trained density estimators on the testing data in the SR. This consists of the second half of the background events plus some number of signal events.

python eval_LHCORD_withsig.py --innermodel=INNERMODEL --outermodel=OUTERMODEL

This produces a number of npy files:

outermodel+'_logPfull_actual' -- the SR density estimate

outermodel+'_logPbg_actual'

outermodel+'_logPfull'

outermodel+'_logPbg'


which are the logP values for each event
