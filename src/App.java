import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
//import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.*;
import com.opencsv.*;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

public class App extends JFrame {

    public static String PATH_TRAIN = "C:\\Users\\alexs\\OneDrive\\Documentos\\Java\\IA\\AI\\Trab.IA\\training\\";
    public static String PATH_TEST = "C:\\Users\\alexs\\OneDrive\\Documentos\\Java\\IA\\AI\\Trab.IA\\teste\\";
    private static final String CSV_PATH_TRAIN = "C:\\Users\\alexs\\OneDrive\\Documentos\\Java\\IA\\AI\\Trab.IA\\csv\\Training.csv";
    private static final String CSV_PATH_TEST = "C:\\Users\\alexs\\OneDrive\\Documentos\\Java\\IA\\AI\\Trab.IA\\csv\\Test.csv";

    private static int[] getPixelData(BufferedImage img, int x, int y) {
        int argb = img.getRGB(x, y);

        int rgb[] = new int[] {
                (argb >> 16) & 0xff, // red
                (argb >> 8) & 0xff, // green
                (argb) & 0xff // blue
        };

        return rgb;
    }

    // ler matrizes da imagem e gerar projeção
    public static void getImagesProjection(int numImages, int projectionSize, String path) throws IOException {
        BufferedImage img;
        int[][] rmat = null;
        int[][] gmat = null;
        int[][] bmat = null;
        List<String[]> data = new ArrayList<String[]>();

        String[] headers = new String[projectionSize + 1];
        for (int i = 0; i < headers.length - 1; i++) {
            headers[i] = "p_" + i;
        }
        headers[headers.length - 1] = "class";

        data.add(headers);

        //String nomeClass = "a";

        char[] classesTest = { 'a', 'o', 'o', 'a', 'o', 'o', 'o', 'o', 'o', 'a', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'a',
                'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o' };
        char[] classesTrain = { 'a', 'a', 'a', 'a', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o',
                'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o' };

        for (int cont = 1; cont <= numImages; cont++) {

            try {
                img = ImageIO.read(new File(path + cont + ".jpg"));

                rmat = new int[img.getHeight()][img.getWidth()];
                gmat = new int[img.getHeight()][img.getWidth()];
                bmat = new int[img.getHeight()][img.getWidth()];

                for (int i = 0; i < img.getHeight(); i++) {
                    for (int j = 0; j < img.getWidth(); j++) {
                        rmat[i][j] = getPixelData(img, j, i)[0];
                        gmat[i][j] = getPixelData(img, j, i)[1];
                        bmat[i][j] = getPixelData(img, j, i)[2];
                    }
                }

            } catch (IOException e) {
                e.printStackTrace();
            }
            int[][] binaryMat = new int[bmat.length][bmat[0].length];
            // imagem binária e invertida em cores
            for (int l = 0; l < bmat.length; l++) {
                for (int c = 0; c < bmat[0].length; c++) {
                    int mean = (bmat[l][c] + rmat[l][c] + gmat[l][c]) / 3;
                    binaryMat[l][c] = mean > 167 ? 0 : 1;
                    System.out.print(binaryMat[l][c]);
                    
                }
                System.out.println();
            }
            // algoritmo da projeção:
            int[] projectionAux = new int[binaryMat[0].length];// tamanho do numero de colunas
            for (int c = 0; c < binaryMat[0].length; c++) {
                for (int l = 0; l < binaryMat.length; l++) {
                    projectionAux[c] += binaryMat[l][c];
                }
            }

            // compacta vetor de projeção:
            int[] projection = new int[projectionSize];
            int move = projectionAux.length / projectionSize;// binaryMat.length/projectionSize;
            int offset = 0;

            for (int c = 0; c < projection.length; c++) {
                for (int c2 = 0; c2 < move; c2++) {
                    projection[c] += projectionAux[offset + c2];
                }
                projection[c] = projection[c] / move;// para virar uma média
                offset += move;
            }

            // imprime projecao

            for (int c = 0; c < projection.length; c++)
                System.out.print(projection[c] + ",");
            System.out.println("");

            if (path.equals(PATH_TEST)) {

                char classTest = classesTest[cont - 1];

                creatCsv(projection, data, classTest, path);

            } else {

                char classTrain = classesTrain[cont - 1];

                creatCsv(projection, data, classTrain, path);
            }
        }

    }

    public static void creatCsv(int[] vetor, List<String[]> data, char nomeClass, String path) throws IOException {
        System.out.println("Iniciando geração");

        // String[] vetorString = Arrays.toString(vetor).split("[\\[\\]]")[1].split(",
        // ");
        String[] vetorString = new String[vetor.length + 1];

        for (int i = 0; i < vetor.length; i++) {
            vetorString[i] = Integer.toString(vetor[i]);
        }

        if (path.equals(PATH_TEST)) {

            FileWriter fwTest = new FileWriter(new File(CSV_PATH_TEST));
            CSVWriter cwTest = new CSVWriter(fwTest);

            vetorString[vetorString.length - 1] = nomeClass + "";

            data.add(vetorString);

            cwTest.writeAll(data);
            cwTest.close();
            fwTest.close();
        } else {

            FileWriter fw = new FileWriter(new File(CSV_PATH_TRAIN));
            CSVWriter cw = new CSVWriter(fw);

            vetorString[vetorString.length - 1] = nomeClass + "";

            data.add(vetorString);

            cw.writeAll(data);
            cw.close();
            fw.close();
        }

    }

    public static void convertCsvToArff(String csvPath, String arffPath) throws IOException {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(csvPath));
        Instances data = loader.getDataSet();

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(arffPath));
        saver.writeBatch();
    }

    /**
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {

        getImagesProjection(32, 20, PATH_TRAIN);
        getImagesProjection(32, 20, PATH_TEST);

        String arffPath = "C:\\Users\\alexs\\OneDrive\\Documentos\\Java\\IA\\AI\\Trab.IA\\arff\\train.arff";
        convertCsvToArff(CSV_PATH_TRAIN, arffPath);

        DataSource dataSource = new DataSource(
                "C:\\Users\\alexs\\OneDrive\\Documentos\\Java\\IA\\AI\\Trab.IA\\arff\\train.arff");

        Instances dataTrain = dataSource.getDataSet();

        dataTrain.setClassIndex(dataTrain.numAttributes() - 1);

        // Train decision tree
        SMO svm = new SMO();
        svm.buildClassifier(dataTrain);

        MultilayerPerceptron rna = new MultilayerPerceptron();
        rna.buildClassifier(dataTrain);

        // Testando
        convertCsvToArff(CSV_PATH_TEST,
                "C:\\Users\\alexs\\OneDrive\\Documentos\\Java\\IA\\AI\\Trab.IA\\arff\\test.arff");
        dataSource = new DataSource("C:\\Users\\alexs\\OneDrive\\Documentos\\Java\\IA\\AI\\Trab.IA\\arff\\test.arff");

        Instances dataTest = dataSource.getDataSet();
        dataTest.setClassIndex(dataTest.numAttributes() - 1);

        // Validando
        Evaluation evaluationSvm = new Evaluation(dataTrain);
        evaluationSvm.evaluateModel(svm, dataTest);
        Evaluation evaluationRna = new Evaluation(dataTrain);
        evaluationRna.evaluateModel(svm, dataTest);

        // analisando SVM
        System.out.println("=======================================SVM=======================================");
        System.out.println(evaluationSvm.toSummaryString());
        System.out.println(evaluationSvm.toClassDetailsString());
        System.out.println(evaluationSvm.toMatrixString());

        // analisando RNA
        System.out.println("=======================================RNA=======================================");
        System.out.println(evaluationRna.toSummaryString());
        System.out.println(evaluationRna.toClassDetailsString());
        System.out.println(evaluationRna.toMatrixString());

    }
}