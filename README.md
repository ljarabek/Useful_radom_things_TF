# Useful_radom_things_TF
Random personally useful things in Tenosrflow to avoid writing same code multiple times



## tf_IO.py:

save(): makes .pickle file with current session weights.

load(): assigns variable vaiues in session with saved weights if they are in .pickle file

This is better for experimentation then checkpoints (and saved_models by extension) because you can freely edit the model and load() will restore weights that were already optimized.

## Interpolation_TF.py

Tensorflow (keras does) doesn't have interpolations for volumetric 3d images. They are implemented as special cases of transpose 3d convolutions. The interpolations in the file halve/double dimensions of the image:

interpolation_linear(): linear interpolation
interpolation_NN(): nearest-neighbor interpolation

## Fading_connections.py

With the advent of progressively growing architectures and obvious benifits of using them, one finds very useful to make any experimental architecture progressively growing to improve inference performance. Especially in image generation tasks, the reconstructions are much sharper.

Alphas(): interchangeably turn on first N layers connections 0-1, keep Nth layer on 1

Betas(): Progressively turning on and fading across N layers, starting with 0th, but never fading Nth
