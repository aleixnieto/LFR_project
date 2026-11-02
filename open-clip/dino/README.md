
# DINO implementation
We use DINO to adapt the representations of CLIP's visual encoder into remote sensing images

## Background on DINO

### Intro
DINO is a method for training neural networks on data without labels. It consists of a teacher-student architecture where the student and teacher are initialized as copies of another. 

### Data
The input data consists of augmentations from the same input data x. 
- student: 2 global crops + 8 local crops
- teacher: 2 global crops
where:
- global crop: image area is 40-100% of the original image
- local crop: image area is 5 to 40% of the original image

### Loss
Assuming there are 2 global crops and 8 local crops, the loss is defined as:
$$
L = \frac{1}{18}*\sum_{i=1}^{2}\sum_{j=1,j\neqi}^{10}p_{t}^{(i)}*\log(p_{s}^{(j)}).
$$

### Training
The student is updated with back-propagation, while the teacher is a moving average of the student:

$$
\theta_t = m*\theta_t + (1-m)*\theta_s.
$$

### Centering
The center is updated with momentum as well:

$$
c = m*c + (1-m) \frac{1}{B}\sum_{i=1}^{B}g_{\theta_t}(x_i)
$$

### Augmentation
See DINO.

## How we do it
We initialize the student and the teacher as copies of OpenCLIP's visual encoder, each with an attached MLP (3 linear layers interscaled with GELU activation functions). The head is discarded after training. We train the networks on the unlabeled images of the training data for 30 epochs.

To know when to stop training, we evaluate the representations on the labeled validation data every 5th epoch. 

We use training parameters from DINO:
- out_dim, hidden_dim, bottleneck_dim
- teacher_momentum, center_momentum, student_temperature, teacher_temperature
- global_crops_scale, local_crops_scale, local_crops_number
- weight_decay

Not specified in DINO:
- encoder_lr, head_lr

## Run it yourself
`python open_clip/dino/main_dino.py`

## Results
[Training results](dino/training_history.png).

