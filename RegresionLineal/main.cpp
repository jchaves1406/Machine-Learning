#include "extoeigen.h"
#include "modellinealregression.h"

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>

/* Se requiere crear una aplicación que lea ficheros que contengan set de datos en CSV
 * (dataset), debe presentar una clase que represente la abstracción de los datos la
 * carga de los datos, la normalizacion de los datos y la manipulacion de los datos
 * con Eigen: ExToEigen
 *
 * Adicional se requiere otra clase que presente el calculo de la regresion lineal.
 */

int main(int argc, char *argv[])
{
    /* Se crea un objeto de tipo ExToEigen, en el cual se incluyen los tres argumentos
     * requeridos por el constructor (NombreDataSet, Delimitador, Header) */
    //
    extoeigen extraccion(argv[1], argv[2], argv[3]);
    ModelLinealRegression LR;

    /* A continuacion se leen los datos del fichero por la funcion leerCSV */
    std::vector<std::vector<std::string>> conjuntoDatos = extraccion.LeerCSV();

    /* Para probar la segunda funcion se define la cantidad de filas y columnas basados
     * en los datos de entrada. */
    int filas = conjuntoDatos.size() + 1;
    int columnas = conjuntoDatos[0].size();
    
    /* Elaboración de matrices */
    Eigen::MatrixXd matrizDF = extraccion.CSVtoEigen(conjuntoDatos, filas, columnas);
    Eigen::MatrixXd difPromedio = matrizDF.rowwise() - extraccion.Promedio(matrizDF);
    Eigen::MatrixXd matrixNorm = extraccion.Normalizacion(difPromedio);

    /* Se requiere verificar la funcion TrainTestSplit si la division de numero de filas y columnas
     * es el esperado para todos los conjuntos de datos (X_train, y_train, x_test, y_test) */
    Eigen::MatrixXd X_train, y_train, x_test, y_test;
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> divDatos = extraccion.TrainTestSplit(matrixNorm, 0.8);

    /* Se desempqueta la tupla del objeto: https://www.cplusplus.com/reference/tuple/tuple/ */
    std::tie(X_train, y_train, x_test, y_test) = divDatos;

    /* Se crean dos vectores para prueba y entrenamiento respectivamente en unos para probar el
     * modelo de regresion lineal */
    Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(x_test.rows());

    /* Redimension de la matriz para ubixar en el vector */
    X_train.conservativeResize(X_train.rows(), X_train.cols() + 1);
    X_train.col(X_train.cols() - 1) = vectorTrain;

    x_test.conservativeResize(x_test.rows(), x_test.cols() + 1);
    x_test.col(x_test.cols() - 1) = vectorTest;

    /* Se define el vector theta inicial como vector de ceros para pasarlo a la funcion del
     * gradiente descendiente. */
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());
    float alpha = 0.01;
    int iteraciones = 1000;

    /* Se definen las variables de salida que representan los coeficientes y el vector de
     * la funcion de costo */
    Eigen::VectorXd thetaOut;
    std::vector<float> costo;

    std::tuple<Eigen::VectorXd, std::vector<float>> gradienteD = LR.GradienteDescendiente(X_train, y_train, theta, alpha, iteraciones);
    std::tie(thetaOut, costo) = gradienteD;

    /* Se imprimen los coeficientes */
    std::cout << thetaOut << std::endl;

    for(auto valor: costo){
        std::cout << valor << std::endl;
    }

    /* Se exportan los valores a ficheros */
    extraccion.VectorToFile(costo, "costo.txt");
    extraccion.EigenToFile(thetaOut, "thetaOut.txt");

    /* Se calcula a base de predicciones el entrenamiento */
    auto mu_data = extraccion.Promedio(matrizDF);
    auto mu_features = mu_data(0, 11);
    auto escalado = matrizDF.rowwise() - matrizDF.colwise().mean();
    auto sigmaData = extraccion.DesviacionEstandar(escalado);
    auto sigmaFeatures = sigmaData(0, 11);
    Eigen::MatrixXd y_train_hat = (X_train * thetaOut * sigmaFeatures).array() + mu_features;
    Eigen::MatrixXd y = matrizDF.col(11).topRows(1279);

    /* Se crea la variable que representa r2 */
    float R2 = LR.R2Cuadrado(y, y_train_hat);

    std::cout << R2 << std::endl;

    extraccion.EigenToFile(y_train_hat, "y_Train_Hat.txt");

    return EXIT_SUCCESS;
}
