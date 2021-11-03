# cnn-camera-calibration
Taking a stab at comma.ai's camera calibration challenge using both
OpenCV and a pretrained EfficientNet model in PyTorch:

<img src="vp_opencv.gif" alt="OpenCV Vanishing Point" width="300px"><img src="vp_efficientnet.gif" alt="Efficientnet Vanishing Point" width="300px">

As I'm writing this the chip shortage is going on (have we really
reached peak compute in 2021???), so I don't currently have easy
access to a stack of 3090's. For that reason, I was a bit sceptical
initially of taking the deep neural network route, especially for hd
video (the example videos have a resolution of 1164x874). However, the
dataset is not very big and pitch and yaw could in principle also be
extracted from individual frames by leveraging parallel lines in
direction of travel (i.e., lane lines) and some rudimentary lane line
detection should be possible using "traditional" computer vision
methods, e.g., available in OpenCV. Parallel lines in frame coordinates
should yield vectors parallel to the Z axis after unprojection into
camera space and deviation in their angles would solve for pitch and yaw.

Unprojection into camera space requires the full camera matrix
(intrinsics K, extrinsics R, and position t), and in this case we only
know K (by knowing width, height, and focal length of the camera).  So
we have to make some assumptions. If we assume a roll of 0, and camera
and world frame to be identical we can solve for pitch and yaw by
unprojecting any point on the unprojected lane lines. To stabilize our
measurements we can choose a particular point that is shared by many
lines. For instance, the point where lines on the left and on the
right cross in the projected image is called the vanishing
point. Unprojecting the vanishing point should yield a vector that
points from the center of the camera in the direction of travel in
camera coordinates (although there are probably other options for
obtaining that motion vector).

I implemented a relatively straightforward approach using OpenCV's
edge detectors on top of the horizontal component in the optical flow
between frames.  Combining the points from those functions into a
stable signal for lane lines yielded some results, but they were very
noisy within a single frame and also unstable across frames. However,
I noticed that results always improved slightly when focusing on the
lower left quadrant of the frame.. this makes sense intuitively, since
that quadrant contains most of what the driver sees (for countries
driving on the wrong side of the road the lower right quadrant would
be a better signal).  Cutting out the sides and filtering the
resulting lane lines using some expected range of angles got me within
the 30% error margin for the labeled examples.

Out of curiosity I also experimented with a Machine-Learning-based
approach using a pre-trained EfficientNet model architecture.  Instead
of regressing pitch and yaw directly from the provided label set per
frame I opted to convert them into pixel coordinates of the vanishing
point in the frame and use a region of interest around the center of
the frame as classification targets in the x and y direction, so that
individual classes represent coordinates.  Since the resulting class
distribution is still very noisy I separated the model to train x and y
coordinates disjointly, using resampling to get a roughly uniform
distribution across all coordinates in the region of interest.  I
trained the pitch and yaw models to ~30 epochs each and then finetuned
on the actual label distribution from the training set for about 15 more
epochs. All in all this took roughly 6 hours total on my mobile GTX 1070
with 8G of VRAM. I trained and finetuned on all labeled files using a
60-20-20 split. At that point the validation accuracy reaches around 60%
with a mean error of around 1.5 pixels.

This means the models will on average predict an area of around 3
pixels around the ground truth vanishing point. However, the dataset
is rather small for a big neural network. While I don't have labels
for the challenge videos, one can verify the predictions to some
extent by overlaying the predicted point on top of the unlabeled
video as I've done in the videos above. While for some of the frames
model predictions are spot on, there are often several off points that
are favored by the network. My guess is that a more diverse set of
training inputs would be required to really make this approach generalize
well, because although using data augmentation methods from Torchvision's
set of random visual transforms can help to diversify the input images,
the label domain of the training set is rather sparsely populated. On the
training set the error maring is around 5%, however, it is unsurprising
for a big neural network with many millions of parameters to be able to
overfit a dataset like this - so, generalization really is the key metric
and in comparison to the OpenCV-based predictions I am not sure that it
actually performs better with the limited training set provided here.

Nonetheless, this challenge was a lot of fun and I did learn a lot along
the way, so kudos to the comma team for putting together this challenge.