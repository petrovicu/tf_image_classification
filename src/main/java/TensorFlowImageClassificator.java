import org.tensorflow.*;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/**
 * Created by petrovicu on 12.4.17..
 */
public class TensorFlowImageClassificator {

    public static void main(String[] args) {

        String imageFile = TensorFlowImageClassificator.class.getResource("jack.jpg").getPath();
        classifyImage(imageFile);
        System.out.println("Labeling result: " + labelImage(imageFile, 5));

    }

    /**
     * Labels an input image by returning a String object with @labelsCount results
     *
     * @param imagePath
     * @param labelsCount
     * @return
     */
    private static String labelImage(String imagePath, int labelsCount) {
        ClassificationResult cr = tensorFlowImageClassification(imagePath);
        float[] labelProbabilities = cr.getLabelProbabilities();
        List<String> labels = cr.getLabels();

        String labelingResult = new String();

        if (labelProbabilities != null) {
            int[] topHitsIndices = indexesOfTopElements(labelProbabilities, labelsCount);
            for (int i : topHitsIndices) {
                System.out.println(
                        String.format(
                                "TOP %d BEST MATCHES: %s (%.2f%% likely)",
                                labelsCount, labels.get(i), labelProbabilities[i] * 100f));
                labelingResult = labelingResult.isEmpty() ? labelingResult : labelingResult.concat(", ");
                labelingResult = labelingResult.concat(labels.get(i));
            }
        } else {
            System.out.println("Error during classification");
        }

        return labelingResult;
    }

    /**
     * Classify image using TensorFlow classificator (inception v5 model)
     *
     * @param imagePath
     * @return
     */
    private static void classifyImage(String imagePath) {
        ClassificationResult cr = tensorFlowImageClassification(imagePath);
        float[] labelProbabilities = cr.getLabelProbabilities();
        List<String> labels = cr.getLabels();

        if (labelProbabilities != null) {
            int bestLabelIdx = maxIndex(labelProbabilities);
            System.out.println(
                    String.format(
                            "BEST MATCH: %s (%.2f%% likely)",
                            labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
        } else {
            System.out.println("Error during classification");
        }
    }

    /**
     * Returns indices of top @nummax probability results
     *
     * @param orig
     * @param nummax
     * @return
     */
    static int[] indexesOfTopElements(float[] orig, int nummax) {
        float[] copy = Arrays.copyOf(orig, orig.length);
        Arrays.sort(copy);
        float[] honey = Arrays.copyOfRange(copy, copy.length - nummax, copy.length);
        int[] result = new int[nummax];
        int resultPos = 0;
        for (int i = 0; i < orig.length; i++) {
            float onTrial = orig[i];
            int index = Arrays.binarySearch(honey, onTrial);
            if (index < 0) continue;
            result[resultPos++] = i;
        }
        return result;
    }

    /**
     * ConstructAndExecuteGraphToNormalizeImage
     * <p>
     * Scale and normalize an image (to 224x224), and convert to a tensor
     *
     * @param imageBytes
     * @return
     */
    private static Tensor constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
        try (Graph g = new Graph()) {
            GraphBuilder b = new GraphBuilder(g);
            // Some constants specific to the pre-trained model at:
            // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
            //
            // - The model was trained with images scaled to 224x224 pixels.
            // - The colors, represented as R, G, B in 1-byte each were converted to
            //   float using (value - Mean)/Scale.
            final int H = 224;
            final int W = 224;
            final float mean = 117f;
            final float scale = 1f;

            // Since the graph is being constructed once per execution here, we can use a constant for the
            // input image. If the graph were to be re-used for multiple input images, a placeholder would
            // have been more appropriate.
            final Output input = b.constant("input", imageBytes);
            final Output output =
                    b.div(
                            b.sub(
                                    b.resizeBilinear(
                                            b.expandDims(
                                                    b.cast(b.decodeJpeg(input, 3), DataType.FLOAT),
                                                    b.constant("make_batch", 0)),
                                            b.constant("size", new int[]{H, W})),
                                    b.constant("mean", mean)),
                            b.constant("scale", scale));
            try (Session s = new Session(g)) {
                return s.runner().fetch(output.op().name()).run().get(0);
            }
        }
    }

    private static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                 Tensor result = s.runner().feed("input", image).fetch("output").run().get(0)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                return result.copyTo(new float[1][nlabels])[0];
            }
        }
    }

    /**
     * Return the classification result with the highest probability for input image
     *
     * @param probabilities
     * @return
     */
    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    private static List<String> readAllLinesOrExit(Path path) {
        try {
            return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;
    }

    /**
     * TensorFlow classificator
     *
     * @param imagePath
     * @return
     */
    private static ClassificationResult tensorFlowImageClassification(String imagePath) {
        String modelDir = TensorFlowImageClassificator.class.getResource("models/inception5h").getPath();

        //Parse model, and read all bytes or exit
        byte[] graphDef = readAllBytesOrExit(Paths.get(modelDir, "tensorflow_inception_graph.pb"));

        //Extract all labels
        List<String> labels =
                readAllLinesOrExit(Paths.get(modelDir, "imagenet_comp_graph_label_strings.txt"));

        //Read image used for classification
        byte[] imageBytes = readAllBytesOrExit(Paths.get(imagePath));


        float[] labelProbabilities = null;
        //DO THE CLASSIFICATION and compute results
        try (Tensor image = constructAndExecuteGraphToNormalizeImage(imageBytes)) {
            labelProbabilities = executeInceptionGraph(graphDef, image);
        }

        return new ClassificationResult(labels, labelProbabilities);
    }

    // In the fullness of time, equivalents of the methods of this class should be auto-generated from
    // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
    // like Python, C++ and Go.
    static class GraphBuilder {
        private Graph g;

        GraphBuilder(Graph g) {
            this.g = g;
        }

        Output div(Output x, Output y) {
            return binaryOp("Div", x, y);
        }

        Output sub(Output x, Output y) {
            return binaryOp("Sub", x, y);
        }

        Output resizeBilinear(Output images, Output size) {
            return binaryOp("ResizeBilinear", images, size);
        }

        Output expandDims(Output input, Output dim) {
            return binaryOp("ExpandDims", input, dim);
        }

        Output cast(Output value, DataType dtype) {
            return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0);
        }

        Output decodeJpeg(Output contents, long channels) {
            return g.opBuilder("DecodeJpeg", "DecodeJpeg")
                    .addInput(contents)
                    .setAttr("channels", channels)
                    .build()
                    .output(0);
        }

        Output constant(String name, Object value) {
            try (Tensor t = Tensor.create(value)) {
                return g.opBuilder("Const", name)
                        .setAttr("dtype", t.dataType())
                        .setAttr("value", t)
                        .build()
                        .output(0);
            }
        }

        private Output binaryOp(String type, Output in1, Output in2) {
            return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0);
        }
    }
}
