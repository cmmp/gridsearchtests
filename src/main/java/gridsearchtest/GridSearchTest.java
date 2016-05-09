package gridsearchtest;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Random;

import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.GridSearch;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SelectedTag;

public class GridSearchTest {

    private static Random rng = new Random(42);
    private static final int NUM_THREADS = 4;

    public J48 searchParams() throws Exception {
        String filePath = "/UCI/diabetes.arff";

        BufferedReader reader = new BufferedReader(
                new InputStreamReader(this.getClass().getResourceAsStream(filePath)));
        Instances data = new Instances(reader);
        reader.close();

        data.setClassIndex(data.numAttributes() - 1);

        J48 j48 = new J48();
        FilteredClassifier fc = new FilteredClassifier();
        fc.setClassifier(j48);

        GridSearch gridSearch = new GridSearch();
        gridSearch.setClassifier(fc);

        gridSearch.setEvaluation(new SelectedTag(GridSearch.EVALUATION_WAUC, GridSearch.TAGS_EVALUATION));

        gridSearch.setXProperty("classifier.confidenceFactor");
        gridSearch.setXMin(0.05);
        gridSearch.setXMax(0.5);
        gridSearch.setXStep(0.05);
        gridSearch.setXBase(10);
        gridSearch.setXExpression("I");

        gridSearch.setYProperty("classifier.minNumObj");
        gridSearch.setYMin(2);
        gridSearch.setYMax(10);
        gridSearch.setYStep(1);
        gridSearch.setYBase(10);
        gridSearch.setYExpression("I");

        gridSearch.setDebug(true);

        gridSearch.setSeed(rng.nextInt());

        gridSearch.setGridIsExtendable(true);
        gridSearch.setNumExecutionSlots(NUM_THREADS);

        J48 bestClassifier = null;

        try {
            gridSearch.buildClassifier(data);
            FilteredClassifier fcResult = (FilteredClassifier) gridSearch.getBestClassifier();
            bestClassifier = (J48) fcResult.getClassifier();
            if (bestClassifier == null)
                System.out.println("got a null best classifier!!!");
        } catch (IllegalStateException e) {
            System.err.println("got illegal state");
            bestClassifier = new J48();
        }

        System.out.println("Best Parameters found: ");
        System.out.println(String.format("conf factor: %.2f", bestClassifier.getConfidenceFactor()));
        System.out.println(String.format("minNumObj: %d", bestClassifier.getMinNumObj()));

        bestClassifier.buildClassifier(data);

        return bestClassifier;
    }

    public static void main(String[] args) throws Exception {
        new GridSearchTest().searchParams();
    }
}
