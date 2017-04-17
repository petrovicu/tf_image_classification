import java.util.List;

/**
 * Created by petrovicu on 17.4.17..
 */
public class ClassificationResult {

    private List<String> labels;
    private float[] labelProbabilities;

    public ClassificationResult(List<String> labels, float[] labelProbabilities) {
        this.labels = labels;
        this.labelProbabilities = labelProbabilities;
    }

    public List<String> getLabels() {
        return labels;
    }

    public void setLabels(List<String> labels) {
        this.labels = labels;
    }

    public float[] getLabelProbabilities() {
        return labelProbabilities;
    }

    public void setLabelProbabilities(float[] labelProbabilities) {
        this.labelProbabilities = labelProbabilities;
    }
}
