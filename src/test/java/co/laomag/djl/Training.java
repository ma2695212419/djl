package co.laomag.djl;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;

import java.awt.*;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * 在训练中，对训练数据进行多次传递（或 epoch），试图找到数据中的模式和趋势，然后将其存储在模型中。
 * 在此过程中，使用验证数据评估模型的准确性。 该模型根据每个时期的发现进行更新，从而提高了模型的准确性。
 * djl.test
 */
public final class Training {

    // 表示模型更新前处理的训练样本数
    private static final int BATCH_SIZE = 32;

    // 通过完整数据集的次数
    private static final int EPOCHS = 2;

    /**
     * The entry point of application.
     *
     * @param args the input arguments
     * @throws IOException        the io exception
     * @throws TranslateException the translate exception
     */
    public static void main(String[] args) throws IOException, TranslateException {
        // 保存模型的位置
        Path modelDir = Paths.get("build/pytorch_models");

        //从目录创建 ImageFolder 数据集
        ImageFolder dataset = initDataset("ut-zap50k-images-square");
        // 将数据集拆分为训练数据集并验证数据集
        RandomAccessDataset[] datasets = dataset.randomSplit(8, 2);

        // 设置损失函数，旨在最大限度地减少错误损失函数根据正确答案（在训练期间）
        // 评估模型的预测，较高的数字是不好的
        // - 意味着模型表现不佳；表示更多错误；想要最小化错误（损失）
        Loss loss = Loss.softmaxCrossEntropyLoss();

        // 设置训练参数（即超参数）
        TrainingConfig config = setupTrainingConfig(loss);

        try (Model model = DjlApplicationTests.Models.getModel(); // empty model instance to hold patterns
             Trainer trainer = model.newTrainer(config)) {
            // metrics collect and report key performance indicators, like accuracy
            trainer.setMetrics(new Metrics());

            Shape inputShape = new Shape(1, 3, DjlApplicationTests.Models.IMAGE_HEIGHT, DjlApplicationTests.Models.IMAGE_HEIGHT);

            // initialize trainer with proper input shape
            trainer.initialize(inputShape);

            // find the patterns in data
            EasyTrain.fit(trainer, EPOCHS, datasets[0], datasets[1]);

            // set model properties
            TrainingResult result = trainer.getTrainingResult();
            model.setProperty("Epoch", String.valueOf(EPOCHS));
            model.setProperty(
                    "Accuracy", String.format("%.5f", result.getValidateEvaluation("Accuracy")));
            model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));

            // save the model after done training for inference later
            // model saved as shoeclassifier-0000.params
            model.save(modelDir, DjlApplicationTests.Models.MODEL_NAME);

            // save labels into model directory
            DjlApplicationTests.Models.saveSynset(modelDir, (List) dataset.getSynset());
        }
    }

    private static ImageFolder initDataset(String datasetRoot)
            throws IOException, TranslateException {
        ImageFolder dataset =
                ImageFolder.builder()
                        // 检索数据
                        .setRepositoryPath(Paths.get(datasetRoot))
                        .optMaxDepth(10)
                        .addTransform(new Resize(DjlApplicationTests.Models.IMAGE_WIDTH, DjlApplicationTests.Models.IMAGE_HEIGHT))
                        .addTransform(new ToTensor())
                        // 随机抽样；不要按顺序处理数据
                        .setSampling(BATCH_SIZE, true)
                        .build();

        dataset.prepare();
        return dataset;
    }

    private static TrainingConfig setupTrainingConfig(Loss loss) {
        return new DefaultTrainingConfig(loss)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }
}