package WineTask;

import Classifier.RandomForest;
import Dataset.DataSet;
import Dataset.Instance;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class WineTask {

    private static final String TRAIN_DATA = "spine.csv";

    /**
     * args[0] = fold or dataset, e.g. train
     * args[1] = fold or dataset, e.g. test
     * args[2] = n_estimators
     * args[3] = max_features = 13
     * args[4] = min_samples_leaf
    **/
    public static void main(String[] args) throws IOException {
        
    	WineTask WineTask = new WineTask();

        int estimators = args.length >= 3 ? Integer.parseInt(args[2]) : 100;
        int maxFeatures = args.length >= 4 ? Integer.parseInt(args[3]) : 8;
        int minSamplesLeaf = args.length >= 5 ? Integer.parseInt(args[4]) : 1;
        
        
        Pair<List<double[]>, List<Integer>> trainData = WineTask.getData(args[0]);
        Pair<List<double[]>, List<Integer>> testData = WineTask.getData(args[1]);
        
        DataSet train = new DataSet(trainData.first, trainData.second);
        DataSet test = new DataSet(testData.first, testData.second);
        
        System.out.println(train.getNumOfInstance());
        System.out.println(train.getLabels());

        RandomForest rf = new RandomForest(new DataSet(train.getTrainingData()), estimators, 3, minSamplesLeaf);
//
        int correct = 0;
        int total = 0;
        for (int i = 0; i < test.getSize(); i++){
            Instance sample = test.getInstance(i);
            int predictLabel = rf.predictLabel(sample.getFeatureVector());
            System.out.println("True Value " + sample.getLabelIndex() + " Predicted Value " + predictLabel);
            if (sample.getLabelIndex() == predictLabel) {
            	correct++;
            }
            total++;
        }

        System.out.println(((double) correct / total));
    }

    private Pair<List<double[]>, List<Integer>> getData(String path) throws IOException {
//        System.out.println("read dataset from " + path);

        try (BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(path)))) {
            int count = -1;
            List<double[]> samples = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();

            String input;
            while ((input = in.readLine()) != null) {
                count++;

                String[] features = input.split(",");
                double[] sample = new double[features.length - 1];
                for (int i = 0; i < features.length - 1; i++){
                    sample[i] = Double.parseDouble(features[i]);
                }
                samples.add(sample);

                labels.add(Integer.parseInt(features[features.length - 1]));
            }
            return new Pair<>(samples, labels);
        }
    }

    private static class Pair<K, V> {
        K first;
        V second;

        public Pair(K first, V second) {
            this.first = first;
            this.second = second;
        }
    }

}
