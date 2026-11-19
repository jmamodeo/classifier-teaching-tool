"""The lesson sequence.

Each section has objectives, a few blocks of explanation, and exercises.
Numbers that the training code also uses are filled in from those constants.
Diagrams are drawn by the app from the block's ``visual`` name; this module
does not import Streamlit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from imagepro.data import (
    FLIP_PROBABILITY,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_ROTATION_DEGREES,
)
from imagepro.training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_VAL_PERCENT,
    LEARNING_RATES,
    MIN_BATCH_SIZE,
)

# Second settings used by exercises. Both sit inside the sidebar's range.
COMPARISON_EPOCHS = 15
COMPARISON_BATCH = 8
LAB_PAGE = "lab"

_INPUTS = 3 * IMAGE_SIZE * IMAGE_SIZE


@dataclass(frozen=True)
class Block:
    """One chunk of a lesson, optionally followed by a named diagram."""

    heading: str
    paragraphs: tuple[str, ...]
    visual: str = ""
    formula: str = ""


@dataclass(frozen=True)
class Exercise:
    """What to change in the training lab, and the check at the end.

    ``changed`` names sidebar settings. Everything else stays at the default
    described by :func:`keep_sentence`.
    """

    changed: tuple[str, ...]
    instructions: str
    look_for: str
    question: str
    answer: str
    key: str = ""


@dataclass(frozen=True)
class Section:
    """One lesson in order."""

    key: str
    number: int
    short: str
    title: str
    objectives: tuple[str, ...]
    blocks: tuple[Block, ...]
    exercise: Exercise
    extra_exercises: tuple[Exercise, ...] = ()

    def exercise_items(self) -> tuple[tuple[str, Exercise], ...]:
        """Return the exercises with their durable checkbox IDs."""
        return (
            (self.key, self.exercise),
            *((f"{self.key}-{item.key}", item) for item in self.extra_exercises),
        )


def default_phrases() -> tuple[tuple[str, str], ...]:
    """Sidebar settings in menu order, phrased as the value the widgets start on."""
    rate = f"{DEFAULT_LEARNING_RATE:.4g}"
    return (
        ("device", "Training device on CPU"),
        ("inference_backends", "all available inference backends checked"),
        ("precision", "export precision on float32"),
        ("mode", "Learning mode on Transfer"),
        ("learning_rate", f"learning rate at {rate}"),
        ("batch_size", f"batch size at {DEFAULT_BATCH_SIZE}"),
        ("epochs", f"epochs at {DEFAULT_EPOCHS}"),
        ("val_percent", f"validation holdout at {DEFAULT_VAL_PERCENT}%"),
        ("shuffle", "shuffle on"),
        ("augment", "augmentation on"),
        ("seed", f"seed at {DEFAULT_SEED}"),
    )


def keep_sentence(changed: Sequence[str]) -> str:
    """Tell the learner which settings to leave alone."""
    known = {key for key, _phrase in default_phrases()}
    unknown = [key for key in changed if key not in known]
    if unknown:
        raise ValueError(f"unknown setting: {', '.join(unknown)}")
    kept = [phrase for key, phrase in default_phrases() if key not in changed]
    if not kept:
        return "Every sidebar setting is part of this exercise."
    joined = "; ".join(kept)
    return f"Leave these at their defaults: {joined}."


def _mean_std() -> tuple[str, str]:
    mean = ", ".join(f"{value:.3f}" for value in IMAGENET_MEAN)
    std = ", ".join(f"{value:.3f}" for value in IMAGENET_STD)
    return mean, std


def _build() -> tuple[Section, ...]:
    mean, std = _mean_std()
    low = f"{LEARNING_RATES[0]:.4g}"
    mid = f"{DEFAULT_LEARNING_RATE:.4g}"
    high = f"{LEARNING_RATES[-1]:.4g}"
    flip = f"{FLIP_PROBABILITY:.0%}"
    degrees = f"{MAX_ROTATION_DEGREES:g}"
    pixels = f"{_INPUTS:,}"
    return (
        Section(
            key="images",
            number=1,
            short="Pixels",
            title="Images as numbers",
            objectives=(
                "Read a picture as a grid of numbers, with three color channels.",
                "State the shape of one image and of a batch in this app.",
                "Explain the resize and the normalization that happen before the network.",
            ),
            blocks=(
                Block(
                    heading="A picture is a grid",
                    paragraphs=(
                        "A digital picture is a grid. Each cell is a pixel. In a color "
                        "picture each pixel holds three integers, usually from 0 to 255: "
                        "one for red, one for green, and one for blue. Those three lists "
                        "are the channels. A grayscale picture would have one channel. "
                        "The drawings below are real built-in shapes, the same kind the "
                        "training lab can learn.",
                        "The heatmaps are the three channels of one drawing. The grid is "
                        "an 8×8 patch of the red channel, chosen because the red values "
                        "there actually vary. A flat patch of background would be a less "
                        "honest picture of the grid.",
                    ),
                    visual="channels",
                ),
                Block(
                    heading="The block of numbers the network multiplies",
                    paragraphs=(
                        "The word tensor, in this app, means that block of numbers. One "
                        f"image is three-dimensional: {3} channels, then height, then width. "
                        f"After resizing, that is 3 × {IMAGE_SIZE} × {IMAGE_SIZE}, which is "
                        f"{pixels} numbers. A batch stacks several images on a new axis, so "
                        f"its shape is N × 3 × {IMAGE_SIZE} × {IMAGE_SIZE}. N is how many "
                        "images are scored together.",
                        "PyTorch stores the channels before the height and the width. A "
                        f"person often thinks of a photo as {IMAGE_SIZE} × {IMAGE_SIZE} × 3. "
                        "The pixels are the same; the order of the axes is a convention the "
                        "multiplications depend on. When a shape is printed later, read it "
                        "as channels first.",
                    ),
                    visual="tensor",
                ),
                Block(
                    heading="Resize, then put the numbers on the scale the network expects",
                    paragraphs=(
                        "Every image is resized to "
                        f"{IMAGE_SIZE}×{IMAGE_SIZE} with a bilinear resize: each new pixel "
                        "is a weighted blend of the pixels around it. A phone photo and a "
                        "tiny drawing become the same shape, which is the shape this "
                        "ResNet was built for.",
                        "The integers are then divided by 255, so each channel sits between "
                        "0 and 1. Then each channel is shifted and scaled by the mean and "
                        "standard deviation of ImageNet, the large set of photographs the "
                        f"pretrained weights saw. The means are {mean}. The standard "
                        f"deviations are {std}. Both modes use those same numbers, so a "
                        "comparison of Transfer and Novel is about the weights, not about "
                        "a different pixel scale.",
                        "Move the slider under the charts. It recomputes one pixel of the "
                        "patch: the 0–255 triple, the 0–1 triple, and the normalized triple.",
                    ),
                    visual="normalize",
                ),
            ),
            exercise=Exercise(
                changed=(),
                instructions=(
                    "In the training lab below, choose Built-in shapes. Leave every "
                    "sidebar setting where it is. Look at the row of pictures under "
                    "Augmentation and the holdout. You do not need to press Train."
                ),
                look_for=(
                    "The training draws are the same drawing, flipped or slightly rotated. "
                    "The validation picture is only resized. Both of them still go through "
                    "the 0–1 scaling and the ImageNet normalization before the network "
                    "sees them; the preview is showing you the pictures, not those numbers."
                ),
                question=(
                    "The validation image skips the flip and the rotation. What still "
                    "happens to its pixels before the network multiplies them, and why "
                    "does prediction use that same path?"
                ),
                answer=(
                    "It is resized, scaled from 0–255 into 0–1, and then normalized with "
                    "the ImageNet mean and standard deviation. The flip and the rotation "
                    "are training-only, so the holdout stays a check on images the "
                    "optimizer did not practice on. Prediction scores a new picture the "
                    "way validation does: resize, scale, and normalize, with no flip and "
                    "no rotation."
                ),
            ),
        ),
        Section(
            key="neuron",
            number=2,
            short="Neuron",
            title="A neuron and a linear layer",
            objectives=(
                "Compute a weighted sum and add a bias.",
                "Describe the final layer of this ResNet: one score per class.",
                "Explain softmax as turning scores into shares that add to 1.",
            ),
            blocks=(
                Block(
                    heading="One neuron is a weighted sum",
                    paragraphs=(
                        "A neuron takes a list of inputs, multiplies each one by its own "
                        "weight, adds those products, and adds one more number called the "
                        "bias. A positive weight means that input pushes the result up. A "
                        "negative weight pushes it down. A zero input contributes nothing, "
                        "whatever its weight is. The picture uses a three-input example "
                        "you can recompute by hand. Drag the bias and the total moves by "
                        "exactly that amount.",
                    ),
                    formula=r"y = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b",
                    visual="neuron",
                ),
                Block(
                    heading="The last layer is many of those, one per class",
                    paragraphs=(
                        "This ResNet ends in a linear layer, which is one neuron per "
                        "class, all of them reading the same inputs. Those inputs are "
                        "what remains after the convolutions and a global average pool: "
                        "one number per channel of the last stage. The layer's weights "
                        "are a matrix with one row per class and one column per input, "
                        "plus one bias per class. Its outputs are scores. A score may be "
                        "negative. It is not yet a share of anything.",
                        "The lines under the chart are read from this model, for the "
                        "class count currently set in the lab. Transfer mode trains this "
                        "layer and freezes everything in front of it. Novel mode trains "
                        "this layer too, and also trains the rest.",
                    ),
                    visual="final_layer",
                ),
                Block(
                    heading="Softmax turns scores into shares",
                    paragraphs=(
                        "Softmax takes the exponential of each score and divides by the "
                        "sum of those exponentials. Every share is positive, and the "
                        "shares add to 1, so they can be drawn as percents. The largest "
                        "score gets the largest share. How large depends on the gaps: "
                        "two equal scores split their part of the pie, and one score far "
                        "above the others takes almost all of it.",
                        "The implementation subtracts the largest score before the "
                        "exponential. That keeps the math from overflowing, and it does "
                        "not change the shares. Move the sliders and watch the shares "
                        "move. The percent beside a prediction, later, is the winning "
                        "share from this same function.",
                    ),
                    formula=r"\mathrm{share}_i = e^{z_i} \big/ \sum_j e^{z_j}",
                    visual="softmax",
                ),
            ),
            exercise=Exercise(
                changed=("mode",),
                instructions=(
                    "Stay on Built-in shapes. Change only Learning mode, from Transfer "
                    "to Novel and back, and read the two parameter cards under the "
                    "images. You do not need to press Train."
                ),
                look_for=(
                    "The total parameter count is the same on both cards. Transfer's "
                    "trainable count is only the new final layer. Novel's trainable "
                    "count equals the total. The final-layer arithmetic in this section "
                    "should match Transfer's trainable count for the class count you "
                    "are looking at (three, for the built-in shapes)."
                ),
                question=(
                    "Transfer's trainable count equals (inputs × classes) + classes. "
                    "Where do the inputs come from, and why is the rest of the network "
                    "missing from that count?"
                ),
                answer=(
                    "The convolutions end in a global average pool that leaves one "
                    "number per channel of the last stage. That width is the input "
                    "count of the final layer; this section prints it from the model. "
                    "Each class gets one weight for each of those numbers, plus one "
                    "bias. In Transfer mode every earlier parameter is frozen, so it "
                    "is in the total and not in the trainable count. The new final "
                    "layer starts with random weights in both modes. Transfer's head "
                    "start is the frozen feature extractor in front of that layer."
                ),
            ),
        ),
        Section(
            key="learning",
            number=3,
            short="Learning",
            title="Loss, gradients, and the learning rate",
            objectives=(
                "Define the cross-entropy loss for one image, and the mean this app plots.",
                "Say what a gradient is, and what a gradient step does with it.",
                "Explain what the learning rate scales, and what Adam changes about the step.",
            ),
            blocks=(
                Block(
                    heading="The loss scores one image",
                    paragraphs=(
                        "After softmax, the correct class has some share p between 0 and "
                        "1. The loss for that image is the negative natural logarithm of "
                        "p, the ln from algebra class. If p is 1, ln(1) is 0, so the loss "
                        "is 0. If p is 1/e, about 0.37, the loss is 1. If p is tiny, the "
                        "loss is large. A confident wrong answer costs more than a near "
                        "miss.",
                        "The chart on the training lab is the mean of this loss over the "
                        "images scored in the epoch: add the per-image losses, divide by "
                        "how many images were scored. It does not grow just because the "
                        "epoch was split into more batches. The network's loss function "
                        "takes the raw scores and applies the logarithm of the softmax "
                        "itself; the value is the same quantity.",
                    ),
                    formula=r"\ell = -\ln(p)",
                    visual="loss",
                ),
                Block(
                    heading="A gradient points uphill; the step walks downhill",
                    paragraphs=(
                        "The loss depends on every trainable weight. The gradient is the "
                        "list of partial derivatives: how much the loss would change if "
                        "you nudged that one weight and held the others still. Gradient "
                        "descent subtracts a fraction of the gradient from the weights. "
                        "The fraction is the learning rate. A small step usually lowers "
                        "the loss. A step that is too large can climb out the other side.",
                        "The diagram is a one-number toy, the bowl (weight − 2)², updated "
                        "by plain gradient descent. It is not Adam, and it is not the "
                        "ResNet. It is there so the step size is visible. Try a small "
                        "step, a step of 0.50 (which lands on the bottom in one update "
                        "from the marked start), and a step of 1.00 (which swaps sides "
                        "of the bowl forever).",
                    ),
                    formula=r"w \leftarrow w - \alpha \nabla \ell(w)",
                    visual="descent",
                ),
                Block(
                    heading="This app steps with Adam, once per batch",
                    paragraphs=(
                        "The optimizer is Adam. Adam keeps a running average of the "
                        "gradient and a running average of the squared gradient, and it "
                        "uses both to rescale the step. The learning rate still "
                        "multiplies that step. The sidebar does not change Adam's other "
                        "settings; the caption under the diagram reads the library "
                        "defaults. Weight decay, an extra pull toward zero, is off.",
                        "The update runs after each batch, not once per epoch. A very "
                        "large learning rate can push a score so far that the loss stops "
                        "being a finite number. The run then stops and asks you to lower "
                        "the learning rate. That message is the toy diagram's overshoot, "
                        "showing up in the real training loop.",
                    ),
                    visual="adam",
                ),
            ),
            exercise=Exercise(
                changed=("learning_rate",),
                instructions=(
                    "Choose Built-in shapes and train three times. Change only the "
                    f"learning rate: first {low}, then {mid}, then {high}. Between runs, "
                    "put every other setting back to its default, including Transfer "
                    f"mode, {DEFAULT_EPOCHS} epochs, and seed {DEFAULT_SEED}. The "
                    "recent-runs table keeps the last few results so you can compare "
                    "them after the next run replaces the charts."
                ),
                look_for=(
                    "The final training loss of each row, and whether a run stopped "
                    "early because the loss was not finite. A smaller step often ends "
                    "these five epochs at a higher loss. A larger step can fall faster, "
                    "or it can diverge. Either outcome is a result. Write down what "
                    "this machine did; the shapes are an easy set, so a large step may "
                    "still finish."
                ),
                question=(
                    "What does raising the learning rate change about the update, and "
                    "which two things does it leave alone?"
                ),
                answer=(
                    "It scales the Adam step taken after each batch. It does not change "
                    "how many epochs run, and it does not change how many images are in "
                    "a batch. Five epochs at two learning rates are the same number of "
                    "passes and a different distance per step. If the loss becomes "
                    "infinite or NaN, the run stops because a step left the region "
                    "where the numbers still make sense."
                ),
            ),
        ),
        Section(
            key="convolutions",
            number=4,
            short="Filters",
            title="Convolutions, depth, and residual connections",
            objectives=(
                "Describe a convolution as a small filter slid across the image.",
                "Explain a feature map, and why stacking layers changes what is visible.",
                "Say what a residual connection adds, and where this ResNet's parameters sit.",
            ),
            blocks=(
                Block(
                    heading="A filter is a small dot product that slides",
                    paragraphs=(
                        "A convolutional layer does not look at the whole image at once. "
                        "It keeps a small grid of weights, the filter, and slides it "
                        "across the picture. At each stop it takes the dot product of the "
                        "filter with the local patch and writes one number. The grid of "
                        "those numbers is a feature map. One layer usually has many "
                        "filters, so it writes many feature maps.",
                        "The filter in the diagram is hand-built. It is a Sobel kernel, "
                        "which responds to a vertical edge: positive on the right of the "
                        "window, negative on the left. It is not a weight from the "
                        "network. The picture under it is a real built-in drawing, "
                        "reduced to luminance. Move the window and the dot product updates "
                        "from the nine products.",
                    ),
                    visual="convolution",
                ),
                Block(
                    heading="Stacking layers, and the skip connection",
                    paragraphs=(
                        "The first filters see small patterns: an edge, a blob of color. "
                        "The next layer's filters look at those feature maps rather than "
                        "at the pixels, so they can respond to a corner or a texture. "
                        "Depth is this stacking. A stride greater than 1 makes the filter "
                        "jump, so the feature map shrinks and later layers see a wider "
                        "piece of the original image in each cell.",
                        "ResNet-18 inserts a residual connection in each block. The block "
                        "is two 3×3 convolutions. Their output is added to the block's "
                        "input, and then a ReLU is applied. When the channel count or the "
                        "spatial size changes, a 1×1 convolution reshapes the input so the "
                        "two sides of the addition match. The block can learn a small "
                        "change. If the useful change is zero, the added path still "
                        "carries the features forward.",
                    ),
                    visual="residual",
                ),
                Block(
                    heading="Where the parameters and the multiply-adds actually are",
                    paragraphs=(
                        "The bars are counted from this ResNet-18, with random weights "
                        "so nothing is downloaded. Parameter counts do not depend on the "
                        "pixel values. A 3×3 convolution has 9 weights for every pair of "
                        "input channel and output channel, plus one bias per output "
                        "channel if that layer has a bias. Later stages have more "
                        "channels, so they hold more weights. The final linear layer is "
                        "a thin slice on top of that.",
                        "The first-layer picture uses the model you last trained, when "
                        "there is one. Otherwise it shows the random filters novel mode "
                        "starts from. Each filter is scaled on its own so you can see "
                        "it; that scaling is for the picture, not a step the network takes.",
                    ),
                    visual="stages",
                ),
            ),
            exercise=Exercise(
                changed=("mode",),
                instructions=(
                    "Train the built-in shapes once in Transfer mode with every other "
                    "setting at its default. Come back to this section and look at the "
                    "first-layer filters. Then change only Learning mode to Novel, "
                    "train again, and look at the filters once more."
                ),
                look_for=(
                    "In Transfer mode the first layer is the one that arrived with the "
                    "ImageNet weights, and this short run does not change it, because "
                    "that layer is frozen. The filters have visible structure. In Novel "
                    "mode the same layer started random and five epochs on these "
                    "drawings do not turn it into clean edge detectors. The parameter "
                    "bars do not change with the mode. They describe the shape of the "
                    "network."
                ),
                question=(
                    "The skip path adds the block's input to the output of its "
                    "convolutions. What does that let the block learn, and what happens "
                    "when the input and the output have different shapes?"
                ),
                answer=(
                    "The block can learn a change to the features instead of having to "
                    "rebuild them. If the useful change is small, the convolutional "
                    "path can stay small and the input still gets through the addition. "
                    "When the number of channels or the height and width differ, a 1×1 "
                    "convolution changes the input's shape first, and that reshaped "
                    "input is what gets added. A 1×1 convolution mixes channels at a "
                    "single pixel; it does not look at neighbors."
                ),
            ),
        ),
        Section(
            key="dynamics",
            number=5,
            short="Epochs",
            title="Epochs, batches, and overfitting",
            objectives=(
                "Define an epoch, a batch, and the holdout.",
                "Recognize overfitting as training loss falling while validation loss does not.",
                "Say what augmentation and the seed change from one run to the next.",
            ),
            blocks=(
                Block(
                    heading="An epoch is a pass; a batch is one update",
                    paragraphs=(
                        "An epoch is one pass through the training images. A batch is "
                        "the group scored before the next Adam step. The strip below "
                        "uses the built-in shapes, the holdout currently in the sidebar, "
                        "and the batch size currently in the sidebar. It is the real "
                        "split, not a sketch: each class keeps at least one training "
                        "image and one validation image, and the holdout is the nearest "
                        "count to the percentage you set.",
                        f"ResNet's batch normalization cannot learn from a single image, "
                        f"so the batch size starts at {MIN_BATCH_SIZE} and a leftover "
                        "batch of one image is left out of that epoch. Shuffle puts the "
                        "training images in a new order every epoch, which keeps a long "
                        "run of one class from dominating several updates in a row. The "
                        "validation images stay in a fixed order.",
                    ),
                    visual="epoch_strip",
                ),
                Block(
                    heading="The holdout is the check",
                    paragraphs=(
                        "Validation images are scored after every epoch. Their loss and "
                        "accuracy never enter the weight update. If they did, a falling "
                        "validation loss would just be more training. The two curves "
                        "share an axis so a gap is visible. On a handful of held-out "
                        "images the numbers jump; read the direction.",
                        "Overfitting is the training loss going on down while the "
                        "validation loss flattens or rises. The model is getting better "
                        "at the images it trains on and not at the ones it doesn't. The "
                        "sketch shows that shape with invented y-values, so you know "
                        "what to look for. It is not a result from this app. More "
                        "images, augmentation, fewer epochs, and Transfer mode, which "
                        "leaves fewer weights free, are the counters this lab actually "
                        "has.",
                    ),
                    visual="overfit",
                ),
                Block(
                    heading="Augmentation and the seed",
                    paragraphs=(
                        f"Each time a training image is loaded, it is flipped left-right "
                        f"with probability {flip} and rotated by up to {degrees} degrees. "
                        "The next epoch loads it again and draws a new flip and angle, "
                        "so the model does not see an identical tensor every pass. "
                        "Validation images, and any picture you classify later, are only "
                        "resized and normalized. The preview in the lab uses its own "
                        "random numbers, so looking at it does not change the run.",
                        "The seed fixes the initial weights, which images are held out, "
                        "the shuffle order, and the augmentation sequence. The same seed "
                        "on a CPU is meant to repeat the run. A different seed shows how "
                        "much the result depended on those choices. Some GPU kernels can "
                        "still differ by a small amount.",
                    ),
                ),
            ),
            exercise=Exercise(
                changed=("epochs", "augment"),
                instructions=(
                    "Use the built-in shapes. First, change only Epochs to "
                    f"{COMPARISON_EPOCHS} and train. Then set Epochs back to "
                    f"{DEFAULT_EPOCHS}, turn Augment training images off, and train "
                    "again. Leave the learning rate, the batch size, the holdout, "
                    "shuffle, the seed, and Transfer mode at their defaults."
                ),
                look_for=(
                    "On the longer run, whether validation loss keeps moving with "
                    "training loss or peels away. On the second run, the preview has "
                    "no extra draws: every epoch sees the same resized pixels. The "
                    "recent-runs table holds both finals. With so few images, treat "
                    "the shape of the curves as the lesson."
                ),
                question=(
                    "Validation is scored every epoch. Why must those images stay out "
                    "of the weight update?"
                ),
                answer=(
                    "They are the check on images the optimizer did not train on. If "
                    "the update is allowed to use them, a falling validation loss no "
                    "longer means the model works past the training set. It would be "
                    "more training data under another name. This holdout is small, so "
                    "the check is noisy. Read the direction of the curve, and treat a "
                    "perfect score on two images as a demonstration."
                ),
            ),
        ),
        Section(
            key="transfer",
            number=6,
            short="Transfer",
            title="Transfer learning and training from scratch",
            objectives=(
                "Contrast where the weights start in Transfer mode and in Novel mode.",
                "Explain a frozen layer: it still runs, and it receives no gradient.",
                "Say what happens to the BatchNorm statistics in each mode.",
            ),
            blocks=(
                Block(
                    heading="Same shape, different starting weights",
                    paragraphs=(
                        "Both modes build a ResNet-18 and replace the last layer with "
                        "one that has your classes. Transfer mode loads weights trained "
                        "on ImageNet, then freezes every parameter except the new final "
                        "layer. Those frozen layers still run on every image. They still "
                        "produce features. They do not receive a gradient, and Adam does "
                        "not step them. The first Transfer run downloads the ImageNet "
                        "weights. The training lab states the file size when that starts.",
                        "Novel mode builds the same architecture with random weights and "
                        "trains every parameter, including the new final layer. The final "
                        "layer starts random in both modes. What changes is the backbone: "
                        "pretrained and frozen, or random and trainable. The images, the "
                        "loss, and the learning rate are the same. The cards under the "
                        "images count both choices for the class count in the lab.",
                    ),
                    visual="modes",
                ),
                Block(
                    heading="BatchNorm keeps a mean and a variance per channel",
                    paragraphs=(
                        "Batch normalization shifts each channel using a mean and a "
                        "variance. During ordinary training those come from the current "
                        "batch, and a running average is stored for later. The stored "
                        "average is what evaluation uses, including validation and "
                        "prediction.",
                        "In Transfer mode the backbone is frozen, but leaving BatchNorm "
                        "in training mode would still let those stored averages drift. "
                        "The app keeps every BatchNorm layer in evaluation mode, so the "
                        "ImageNet mean and variance stay put while the final layer "
                        "trains. That is part of what a frozen backbone means here.",
                        "In Novel mode the running averages start near their "
                        "initialization. The library's default momentum barely moves "
                        "them on a handful of images, and validation would then score "
                        "the features with the wrong shift. After each Novel epoch the "
                        "app recomputes those averages from one pass over the training "
                        "images. The weights of the layer are not changed by that pass. "
                        "They still learn through Adam, like any other trainable parameter.",
                    ),
                    visual="batchnorm",
                ),
            ),
            exercise=Exercise(
                changed=("mode",),
                instructions=(
                    "Use the built-in shapes and seed "
                    f"{DEFAULT_SEED}. Train in Transfer with every other setting at "
                    "its default. Then change only Learning mode to Novel and train "
                    "again. If the recent-runs table already has those two runs, you "
                    "can read them instead of training twice more."
                ),
                look_for=(
                    "The trainable counts, the final losses, and the caption under the "
                    "curves about BatchNorm. Novel updates every parameter, so the "
                    "same five epochs are a different experiment, not a slower copy "
                    "of the Transfer run. The final layer starts random either way."
                ),
                question=("During a Transfer run, do the BatchNorm running statistics move?"),
                answer=(
                    "No. Those layers stay in evaluation mode, so they keep the mean "
                    "and variance that were stored with the ImageNet weights. Novel "
                    "mode does the opposite at the end of each epoch: it recomputes "
                    "the running statistics from the training images. Left on the "
                    "library's default momentum, a small set would leave those "
                    "averages near their initialization, and the validation loss "
                    "would describe that mismatch."
                ),
            ),
        ),
        Section(
            key="evaluation",
            number=7,
            short="Scores",
            title="Accuracy, confusion, and confidence",
            objectives=(
                "Read accuracy, a confusion matrix, and per-class accuracy.",
                "Explain what the softmax percent measures.",
                "Notice what a tiny holdout can and cannot support.",
            ),
            blocks=(
                Block(
                    heading="Accuracy is a fraction",
                    paragraphs=(
                        "Accuracy is the number of images labeled correctly divided by "
                        "the number scored. The chart in the lab plots that fraction "
                        "for the training images and for the holdout, on the same axis "
                        "from 0 to 1. Per-class accuracy is the same fraction inside "
                        "one row: of the held-out images of this class, how many the "
                        "model called by the right name. A class with no images in the "
                        "row would be undefined. This app's split never leaves a class "
                        "out.",
                    ),
                ),
                Block(
                    heading="The grid shows which mix-ups happened",
                    paragraphs=(
                        "A confusion matrix puts the true class on the rows and the "
                        "predicted class on the columns. The diagonal is the images "
                        "that were right. An off-diagonal cell is a particular mistake: "
                        "that many images of the row's class were called the column's "
                        "class. Accuracy collapses all of that into one fraction. The "
                        "grid tells you whether the errors are spread out or all one "
                        "kind of mix-up.",
                        "The first grid is made up, so you can practice the arithmetic "
                        "on numbers that are not pretending to be a run. Under it, if "
                        "you have trained, is the confusion matrix from your latest run.",
                    ),
                    visual="evaluation",
                ),
                Block(
                    heading="The percent is a share of the score",
                    paragraphs=(
                        "The percent next to a prediction is the softmax share of the "
                        "winning class, from the lesson on the linear layer. It is a "
                        "property of this image's scores across the classes you trained. "
                        "It does not measure the chance that the label is right. A model "
                        "trained on a few drawings can put 99% of its share on the wrong "
                        "class. The bar chart shows whether one class owns the share or "
                        "the model is split.",
                        "On two held-out images, 100% per-class accuracy means those two "
                        "drawings were labeled correctly. It is a demonstration of the "
                        "measurement, and a weak claim about the next photograph.",
                    ),
                ),
            ),
            exercise=Exercise(
                changed=(),
                instructions=(
                    "Use the built-in shapes and leave every sidebar setting at its "
                    "default. If the latest run already matches that, read it. "
                    "Otherwise train once. Read the confusion matrix and the per-class "
                    "table. If you have a picture of a circle, a square, or a triangle, "
                    "use Try an image as well."
                ),
                look_for=(
                    "Which cells are off the diagonal, and how many validation images "
                    "each percent in the table is made of. A perfect row on two images "
                    "is those two images. If you classify a new picture, read the bar "
                    "chart of shares next to the winning percent."
                ),
                question=(
                    "The model assigns 95% to the wrong class. What is that 95%, and "
                    "what is 100 minus 95 in this case?"
                ),
                answer=(
                    "95% is the softmax share of the winning class among the classes "
                    "you trained. It is not a measured chance that the label is right, "
                    "and 100 minus 95 is not a measured error rate. It is the share "
                    "left over for the other classes on this image. Use the bar chart "
                    "to see whether that leftover is one close rival or many small "
                    "shares, and use the confusion matrix to see which classes the "
                    "model actually mixed up on the holdout."
                ),
            ),
        ),
        Section(
            key="hardware",
            number=8,
            short="Hardware",
            title="Arithmetic, memory, and the device",
            objectives=(
                "Describe a step as multiply-adds plus memory for weights and activations.",
                "Separate the counts this page computes from the times your run measures.",
                "Relate batch size to images per second and to activation memory.",
                "Compare on-device inference, compute units, and export precision.",
            ),
            blocks=(
                Block(
                    heading="The arithmetic is multiply-adds",
                    paragraphs=(
                        "A convolution and a linear layer are long runs of multiply-adds: "
                        "multiply a weight by an input and add it into a total. The same "
                        "small dot product happens at many pixels and for many filters, "
                        "and those dot products do not have to wait for each other. Bias "
                        "additions, BatchNorm, ReLU, and pooling are real work and much "
                        "smaller than the convolutions, so the count below leaves them out "
                        "and says so.",
                        "FLOPs, on this page, means twice the multiply-adds, one "
                        "multiplication and one addition. Some write-ups call a "
                        "multiply-add itself one FLOP. Both numbers are shown so the "
                        "convention is visible. The count is one forward pass of this "
                        "ResNet, for the class count and the batch size in the sidebar. "
                        "A training step also runs a backward pass, which does more "
                        "arithmetic than the forward pass. That extra work is not in "
                        "the counted number.",
                    ),
                    visual="hardware",
                ),
                Block(
                    heading="What has to sit in memory",
                    paragraphs=(
                        "The parameters are the same tensors no matter how large the "
                        "batch is. Gradients exist for the trainable ones, one buffer "
                        "the same size as those parameters, during the backward pass. "
                        "Adam keeps two more float32 buffers per trainable parameter, "
                        "the running averages it uses to rescale the step. BatchNorm "
                        "also stores a running mean and variance per channel; those "
                        "are buffers, not Adam state.",
                        "Each image also produces activations, the feature maps between "
                        "layers. Their size grows with the batch, because each image "
                        "has its own. The forward-output total adds up the tensor each "
                        "leaf layer returned. Those tensors are not all stored at the "
                        "same instant, so the total is a parts list, not a reading from "
                        "a memory meter. The backward pass keeps extra tensors. Transfer "
                        "mode needs fewer of them, because the frozen backbone receives "
                        "no gradients.",
                        "This app trains in float32, four bytes per number. Export can "
                        "use float16 instead. That choice affects the exported model, "
                        "not the training run.",
                    ),
                ),
                Block(
                    heading="Why a GPU or an accelerator changes the clock",
                    paragraphs=(
                        "A CPU handles many kinds of work. A GPU, reached through CUDA "
                        "or Metal (MPS), is built to do many of these multiplies at once. "
                        "A larger batch gives the GPU more work in parallel; with a tiny "
                        "batch, launching that work can eat up the savings. The GPU "
                        "still needs memory for the extra activations.",
                        "Training changes weights using gradients. You can train on the "
                        "CPU or an available GPU. Inference uses the learned weights "
                        "without changing them. The Neural Engine on Apple silicon is "
                        "built for that kind of work: Core ML can use it for inference, "
                        "but it cannot train the model in this lab.",
                        "The Hardware panel shows what is available and why an option "
                        "might be disabled. "
                        "Images per second in the table is measured wall-clock time: "
                        "training images scored, divided by the training pass, including "
                        "loading the images and the optimizer step. Validation time is "
                        "a separate column. When the run used CUDA, a peak allocator "
                        "figure is reported too. CPU and MPS runs do not invent one.",
                    ),
                    visual="devices",
                ),
                Block(
                    heading="Inference on the device",
                    paragraphs=(
                        "An exported model can classify an image on the device itself. "
                        "The image never needs to leave that device, which helps privacy. "
                        "There is no network round trip to wait for. Dedicated accelerators "
                        "can also do the same work for less energy, though the actual "
                        "latency and energy depend on the machine and the model.",
                        "Core ML can be allowed CPU only, CPU and GPU, CPU and Neural "
                        "Engine, or ALL. These settings say what Core ML may use, not "
                        "what it did use. Timings alone cannot tell you where a layer ran. "
                        "CPU_AND_NE is disabled on machines without a Neural Engine.",
                        "Float16 takes half as many bytes per weight as float32, but the "
                        "package also includes other data. Rounding can move the scores "
                        "without changing the winning class. Use the sidebar Precision "
                        "control and compare measured package size, median latency, "
                        "top-1 agreement, and the largest probability difference "
                        "against float32 PyTorch.",
                    ),
                    visual="on_device",
                ),
            ),
            exercise=Exercise(
                changed=("batch_size", "device"),
                instructions=(
                    f"Use the built-in shapes and batch size {MIN_BATCH_SIZE}. Train on "
                    "CPU, then switch to an available GPU and train with the same "
                    f"settings. Change only the batch size to {COMPARISON_BATCH} and "
                    "repeat both runs. If neither GPU is available, compare the two "
                    "batch sizes on CPU."
                ),
                look_for=(
                    "Find Train s/epoch in the run log by Device and Batch. Did the "
                    "gap change? The first epoch on a GPU can include one-time setup, "
                    "so compare the later epochs in the history as well. In the memory "
                    "chart, forward outputs grow with the batch; parameter and Adam "
                    "bytes stay put."
                ),
                question="Why might a GPU pull ahead on a larger batch but not on a tiny one?",
                answer=(
                    "More images give the GPU more work to do in parallel. On a tiny "
                    "batch, the time to launch that work can cancel out the savings. "
                    "Use your measured times rather than assuming a speedup."
                ),
            ),
            extra_exercises=(
                Exercise(
                    key="neural-engine",
                    changed=("inference_backends",),
                    instructions=(
                        "Train and export a model. Predict whether CPU_AND_NE will beat "
                        "CPU_AND_GPU on one image, then select those backends in Hardware "
                        "and measure them. Without a Neural Engine, CPU_AND_NE is unavailable: "
                        "compare CPU_ONLY and CPU_AND_GPU instead. If Core ML prediction is "
                        "unavailable, compare the PyTorch backends that are enabled."
                    ),
                    look_for="Compare the median latency rows for the same export.",
                    question="Did the option you picked have the lower measured latency?",
                    answer=(
                        "Read the medians on this machine. A lower median means the "
                        "setting was faster here, not that every layer ran on its accelerator."
                    ),
                ),
                Exercise(
                    key="precision",
                    changed=("precision",),
                    instructions=(
                        "Export the same trained model in float32, then switch Precision "
                        "to float16 and export again. Measure each precision with the "
                        "same inference backends. If Core ML prediction is unavailable, "
                        "compare package sizes alone."
                    ),
                    look_for=(
                        "Float16 halves the weight bytes. Check how close the full package "
                        "comes to half the size, whether the winning labels still match "
                        "float32 PyTorch, and how the median latency changes."
                    ),
                    question="Can the scores change even when the predicted labels match?",
                    answer=(
                        "Yes. Rounding can shift probabilities without changing which "
                        "one is largest. Check both the top-1 agreement and the maximum "
                        "absolute probability difference."
                    ),
                ),
                Exercise(
                    key="all",
                    changed=("inference_backends",),
                    instructions=(
                        "Select Core ML ALL and the other available Core ML settings. "
                        "Measure them together on the same export. If Core ML prediction "
                        "is unavailable here, use the check below instead."
                    ),
                    look_for="Which of the other medians is closest to ALL?",
                    question="Which compute unit did ALL pick? What can the table tell you?",
                    answer=(
                        "Similar timings are a clue, not proof. ALL lets Core ML use "
                        "any available unit, and it may split work across them. "
                        "The table reports time, not which unit ran each layer."
                    ),
                ),
            ),
        ),
    )


SECTIONS: tuple[Section, ...] = _build()
SECTION_BY_KEY: dict[str, Section] = {section.key: section for section in SECTIONS}


def visual_names() -> frozenset[str]:
    """Diagram names the lessons ask for."""
    return frozenset(
        block.visual for section in SECTIONS for block in section.blocks if block.visual
    )
