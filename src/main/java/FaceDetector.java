import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

/**
 * Created by petrovicu on 15.4.17..
 */
public class FaceDetector {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        faceDetect("haarcascade_frontalface_alt.xml", "ja.jpg");
    }

    private static void faceDetect(String modelPath, String imagePath) {
        CascadeClassifier faceDetector = new CascadeClassifier(TensorFlowImageClassificator.class.getResource(modelPath).getPath());
        Mat image = Imgcodecs
                .imread(TensorFlowImageClassificator.class.getResource(imagePath).getPath());

        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(image, faceDetections);

        System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));

        for (Rect rect : faceDetections.toArray())
            Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0));

        String filename = "ouput.png";
        System.out.println(String.format("Writing %s", filename));
        Imgcodecs.imwrite(filename, image);
    }
}
