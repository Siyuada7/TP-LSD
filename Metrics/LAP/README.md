# Usage of LAP Metric

## Get the groundtruth, remember all the line labels need first scaled into [0, 128]
`python mat2txt.py`

## Get the prediction data
The prediction mat file need contains the keys: {'lines', 'scores''}
`python mat2txt_pred.py`

## Calculate the LAP
`python cal_LAP.py`
Need change the wire_path && save_file to the result path and save path.

## Plot the PR curves
`python sample_from_npz.py`