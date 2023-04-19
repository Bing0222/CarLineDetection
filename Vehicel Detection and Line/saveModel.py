import pickle
from detection import *


model_pickle = {}
model_pickle['svc'] = svc
model_pickle['scaler'] = X_scaler
model_pickle['orient'] = orient
model_pickle['pix_per_cell'] = pix_per_cell
model_pickle['cell_per_block'] = cell_per_block
model_pickle['spatial_size'] = spatial_size
model_pickle['hist_bins'] = hist_bins
pickle.dump( model_pickle, open( "00-training.pkl", "wb" ))
