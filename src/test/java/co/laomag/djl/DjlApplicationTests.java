package co.laomag.djl;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;

import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;


import java.awt.*;


import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * The type Djl application tests.
 */
@SpringBootTest
class DjlApplicationTests {

    /*@Test
    void contextLoads() {

        String url = "https://github.com/awslabs/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg";
        BufferedImage img = null;
        try {
//            img = BufferedImageUtils.fromUrl(url);
            img = BufferedImageUtils.fromFile(Paths.get("D:\\项目\\djl\\src\\main\\resources\\static\\dog_bike_car.jpg"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        Criteria<BufferedImage, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(BufferedImage.class, DetectedObjects.class)
                        .optFilter("backbone", "resnet50")
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<BufferedImage, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor()) {
                DetectedObjects detection = predictor.predict(img);
                System.out.println(detection);
                saveBoundingBoxImage(img,detection);
            }
        } catch (MalformedModelException e) {
            e.printStackTrace();
        } catch (ModelNotFoundException e) {
            e.printStackTrace();
        } catch (IOException | TranslateException e) {
            e.printStackTrace();
        }
    }

    private static void saveBoundingBoxImage(BufferedImage img, DetectedObjects detection)
            throws IOException {
        Path outputDir = Paths.get("static/MyFile.png");
        System.out.println(outputDir.toString());
        Files.createDirectories(outputDir);

        // Make image copy with alpha channel because original image was jpg
        int numberOfObjects = detection.getNumberOfObjects();

        Graphics2D graphics = img.createGraphics();
        for (int i = 0; i < numberOfObjects; i++) {
            DetectedObjects.DetectedObject item = detection.item(i);
            BoundingBox boundingBox = item.getBoundingBox();
        }
        File f = new File("MyFile.png");

        ImageIO.write(img,"PNG",f);

//        Image newImage = img.duplicate(Image.Type.TYPE_INT_ARGB);
//        newImage.(detection);
//
//        Path imagePath = outputDir.resolve("detected-dog_bike_car.png");
//        // OpenJDK can't save jpg with alpha channel
//        newImage.save(Files.newOutputStream(imagePath), "png");
//        logger.info("Detected objects image has been saved in: {}", imagePath);
    }*/


    /**
     * Test 1.
     */
    @Test
    void Test1(){
        try {
//            DownloadUtils.download("https://djl-ai.s3.amazonaws.com/mlrepo/model/cv/image_classification/ai/djl/pytorch/resnet/0.0.1/traced_resnet18.pt.gz", "build/pytorch_models/resnet18/resnet18.pt", new ProgressBar());
//            ProgressBar progressBar = new ProgressBar();
//            DownloadUtils.download("https://djl-ai.s3.amazonaws.com/mlrepo/model/cv/image_classification/ai/djl/pytorch/synset.txt", "build/pytorch_models/resnet18/synset.txt", progressBar);

//            progressBar.

//            preprocess = transforms.Compose([
//                    transforms.Resize(256),
//                    transforms.CenterCrop(224),
//                    transforms.ToTensor(),
//                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
//])
            Pipeline pipeline = new Pipeline();
            pipeline.add(new Resize(256))
                    .add(new CenterCrop(224, 224))
                    .add(new ToTensor())
                    .add(new Normalize(
                            new float[] {0.485f, 0.456f, 0.406f},
                            new float[] {0.229f, 0.224f, 0.225f}));
            Translator translator = ImageClassificationTranslator.builder()
                    .setPipeline(pipeline)
                    .optApplySoftmax(true)
                    .build();
//            Translator<Image, Classifications> translator = Translator.
            System.setProperty("ai.djl.repository.zoo.location", "build/pytorch_models/resnet18");

            Criteria<Image, Classifications> criteria = Criteria.builder()
                    .setTypes(ai.djl.modality.cv.Image.class, Classifications.class)
                    // only search the model in local directory
                    // "ai.djl.localmodelzoo:{name of the model}"
                    .optArtifactId("ai.djl.localmodelzoo:resnet18")
                    .optTranslator(translator)
                    .optProgress(new ProgressBar()).build();

            try {
                ZooModel model = ModelZoo.loadModel(criteria);
                Path of = Path.of("C:\\Users\\mzp\\Pictures\\Camera Roll\\dog.png");
                ai.djl.modality.cv.Image image = ImageFactory.getInstance().fromFile(of);

//                Image image = BufferedImageFactory.getInstance().fromUrl("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg");
                Predictor<ai.djl.modality.cv.Image, Classifications> predictor = model.newPredictor();
                try {
                    Classifications predict = predictor.predict(image);
                    System.out.println(predict);
                } catch (TranslateException e) {
                    e.printStackTrace();
                }
            } catch (ModelNotFoundException e) {
                e.printStackTrace();
            } catch (MalformedModelException e) {
                e.printStackTrace();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
    // 表示模型更新前处理的训练样本数
    private static final int BATCH_SIZE = 32;

    /**
     * Context loads.
     */
    @Test
    void  contextLoads(){

        ImageFolder dataset =
                ImageFolder.builder()
                        // 检索数据
                        .setRepositoryPath(Paths.get("ut-zap50k-images-square"))
                        .optMaxDepth(20)
                        .addTransform(new Resize(DjlApplicationTests.Models.IMAGE_WIDTH, DjlApplicationTests.Models.IMAGE_HEIGHT))
                        .addTransform(new ToTensor())
                        // random sampling; don't process the data in order
                        .setSampling(BATCH_SIZE, true)
                        .build();

        try {
            dataset.prepare();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (TranslateException e) {
            e.printStackTrace();
        }
    }


    /**
     * The type Models.
     */
    static final class Models {

        /**
         * The constant NUM_OF_OUTPUT.
         */
// the number of classification labels: boots, sandals, shoes, slippers
        public static final int NUM_OF_OUTPUT = 4;

        /**
         * The constant IMAGE_HEIGHT.
         */
// the height and width for pre-processing of the image
        public static final int IMAGE_HEIGHT = 100;
        /**
         * The constant IMAGE_WIDTH.
         */
        public static final int IMAGE_WIDTH = 100;

        /**
         * The constant MODEL_NAME.
         */
// the name of the model
        public static final String MODEL_NAME = "shoeclassifier";

        private Models() {}

        /**
         * Gets model.
         *
         * @return the model
         */
        public static Model getModel() {
            // create new instance of an empty model
            Model model = Model.newInstance(MODEL_NAME);



            // Block is a composable unit that forms a neural network; combine them like Lego blocks
            // to form a complex network
            Block resNet50 =
                    ResNetV1.builder() // construct the network
                            .setImageShape(new Shape(3, IMAGE_HEIGHT, IMAGE_WIDTH))
                            .setNumLayers(50)
                            .setOutSize(NUM_OF_OUTPUT)
                            .build();

            // set the neural network to the model
            model.setBlock(resNet50);
            return model;
        }

        /**
         * Save synset.
         *
         * @param modelDir the model dir
         * @param synset   the synset
         * @throws IOException the io exception
         */
        static void saveSynset(Path modelDir, List synset) throws IOException {
            Path synsetFile = modelDir.resolve("synset.txt");
            try (Writer writer = Files.newBufferedWriter(synsetFile)) {
                writer.write(String.join("\n", (CharSequence) synset));
            }
        }
    }


    /**
     * Main.
     *
     * @param args the args
     * @throws ModelException     the model exception
     * @throws TranslateException the translate exception
     * @throws IOException        the io exception
     */
    public void main(String[] args) throws ModelException, TranslateException, IOException {
        // the location where the model is saved
        Path modelDir = Paths.get("models");

        // the path of image to classify
        String imageFilePath;
        if (args.length == 0) {
            imageFilePath = "ut-zap50k-images-square/Sandals/Heel/Annie/7350693.3.jpg";
        } else {
            imageFilePath = args[0];
        }

        // Load the image file from the path

        ai.djl.modality.cv.Image img = ImageFactory.getInstance().fromFile(Paths.get(imageFilePath));

        try (Model model = Models.getModel()) { // empty model instance
            // load the model
            model.load(modelDir, Models.MODEL_NAME);

            // define a translator for pre and post processing
            // out of the box this translator converts images to ResNet friendly ResNet 18 shape
            ImageClassificationTranslator translator =
                    ImageClassificationTranslator.builder()
                            .addTransform(new Resize(Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT))
                            .addTransform(new ToTensor())
                            .optApplySoftmax(true)
                            .build();

            // run the inference using a Predictor
            try (Predictor<ai.djl.modality.cv.Image, Classifications> predictor = model.newPredictor(translator)) {
                // holds the probability score per label
                Classifications predictResult = predictor.predict(img);
                System.out.println(predictResult);
            }
        }
    }

}
