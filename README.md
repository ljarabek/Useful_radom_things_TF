# Useful_radom_things_TF
Random personally useful things in Tenosrflow to avoid writing same code multiple times



#tf_IO.py:

save(): makes .pickle file with current session weights. 
load(): assigns variable vaiues in session with saved weights if they are in .pickle file

This is better for experimentation then checkpoints (and saved_models by extension) because you can freely edit the model and load() will restore weights that were already optimized to save time.

