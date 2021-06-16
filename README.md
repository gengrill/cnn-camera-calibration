# cnn-camera-calibration
Taking a stab at comma.ai's camera calibration challenge.. (on a budget).

As I'm writing this the chip shortage is going on (have we really
reached peak compute in 2021???), so I don't currently have easy access
to a stack of 3090's. For that reason, I was a bit sceptical initially
of taking the deep neural network route, especially for hd video (the
example videos have a resolution of 1164x874). However, I figured that
yaw and pitch could in principle be extraced from individual frames by
leveraging parallel lines in direction of travel and perpendicular to
the horizon (e.g., using Zhang's method).

In particular, the constrained environment of a road should provide
sufficiently many of those landmarks to allow an unprojection if we make
some assumptions. For instance, assumed parallel lines in frame coordinates
such as lane lines should yield vectors parallel to the Z axis after being
unprojected into camera space and unprojecting perpendicular lines (such as
lightpoles, trees, and signs on the side of the road) should unproject parallel
to the Y axis. Accurate unprojection  requires  the  full  camera
matrix (intrinsics K, extrinsics R, and position t), and in  this  case we
know K (by knowing Width, Height, and Focal length of the camera). Assuming
a roll of  0, the  angle between Z  and unprojected parallel lines (respectively
Y and unprojected perpendiculars) should thus solve for pitch and yaw.

So, I  experimented  with  an  initial approach  based  on  OpenCV's  edge
detectors, goodFeaturesToTrack, as well as calcOpticalFlowFarneback, and tried
combining the points from those functions into a stable signal for perpendicular
and parallel lines. However, the  results  were very noisy within a single frame
and also unstable across frames (although it's entirely possible I picked the
wrong parameters for those functions). However, I noticed that results always
improved slightly when focusing on the lower left quadrant of the frame.. this
makes sense intuitively, since that quadrant contains most of what the driver
sees (for countries driving on the wrong side of the road the lower right
quadrant would be required). I also did not see a huge difference between using
color images and grayscale features (some of the OpenCV functions convert to
grayscale anyways), so a CNN-based approach became feasible after all, even
without using a ton of compute.

I started with a simple LeNet-based CNN, 7 layers total, grayscaled,
cropped, and resized lower left quadrant to 128x128 pixels. At first,
there were a couple of bugs in my OpenCV to Numpy  to Pytorch Tensor
conversions and single batch training helped a lot with figuring these
out. I then noticed that by default vanishing gradients were a problem
with the small target domain, so I rescaled the target domain and also
experimented with different normalizations. Next, I tried removing the
ReLUs in the linear layers and  that  led to  faster  convergence and
less jumpy losses, so now  the decoder part of the model just consists
of straight up linear layers, that regress pitch and yaw directly from
the provided label set per frame.

Scaling up the training from just a single batch the validation loss
was significantly higher if frames were not shuffled so I implemented a
random permutation dataloader that works for multiple video inputs and
will source frames in a batch from all video inputs. For some reason I
started  with a batch  size of  5, but  that  did not  play  well with
frames from 5 different video files, which took me  a while to figure out:
basically, the model struggled to regress 10 values for 5 files, and after
many many epochs it simply appeared to learn classifying which file a
frame belonged to, because predictions for frames would follow the overall
pattern that appears in the label files (e.g., pitch is mostly 3xyaw in one
file). Even though it converged towards  a decreasing total loss, the
individual  predicted  values  would  be way  off for many frames and in
validation tests on unseen inputs  it would just be plain wrong.

Modifying the  batch size to  2 immediately fixed this,  the regressed
values now made more sense  and absolute validation accuracy error was
within +-30%. I also tried out different optimizers, adam takes a good
while  (> 150  epochs) and  the  loss is  very jumpy,  maybe it  would
benefit  from  additional  regularization   or  better  learning  rate
scheduling, but I did not try that. Instead, I ended up using SGD with
momentum  as this  yielded a  huge  improvement and  also allowed  for
higher learning  rates and  much faster  convergence. While  the third
dense layer  now is basically useless  (going from 72 feature  size to
64) I kept it to allow for other batch sizes in principle.

The final model  has about ~4M parameters  (17M on disk), converges to
~0.01MSE  in  about ~20 epochs  (which takes just a few  minutes on my
mobile GTX 1070 with 8G).  I always trained on 4 input files and validated on
the  remaining one for a rough 80-20 split,  repeating the experiment
5 times with a different file for validation each time.  The resulting
error score for the unseen file was between 20% and 30% on average.
While I would be  curious to know how it fares on the unlabeled dataset,
overall this challenge was a lot of fun regardless.
