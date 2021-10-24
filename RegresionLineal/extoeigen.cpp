#include "extoeigen.h"

#include <vector>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>

/* La primera funcion a realizar es la lectura del fichero csv (dataset):
 * es un vector de vectores de tipo string. Leera linea por linea y almacenara
 * en un vector de vectores de tipo string */
std::vector<std::vector<std::string>> extoeigen::LeerCSV(){
    /* Abrir archivo para lectura solamente */
    std::ifstream Fichero(setDatos);

    /* Vector de vectores del tipo string: tendra los datos del dataset */
    std::vector<std::vector<std::string>> datosString;

    /* Se itera a traves de cada linea del dataset, y se divide del contenido
     * usando el delimitador provisto por el constructor. */
    std::string linea = "";

    while(getline(Fichero, linea)){
        std::vector<std::string> vectorFila;
        boost::algorithm::split(vectorFila, linea, boost::is_any_of(delimitador));
        datosString.push_back(vectorFila);
    }

    /* Se cierra el fichero */
    Fichero.close();

    /* Se retorna el vector de vectores de tipo string */
    return datosString;
}

/* Segunda funcion ara guardar el vector de vectores del tipo string
 * (similar a pandas). La idea es presentarlos como un DataFrame(los datos) */

Eigen::MatrixXd extoeigen::CSVtoEigen(std::vector<std::vector<std::string>> setDatos, int filas, int columnas){
    /* Si se tiene cabecera la removemos, se manipula solo los datos (por eso eliminamos la cabecera)
     * si el header es igual a true*/
    if (header == true){
        filas -= 1;
    }

    /* Se itera sobre filas y columnas para almacenar en la matriz vacia de tamaño filas x columnas
     * basicamente almacenara strings en el vector, luego se pasan a flotante (float) para ser manipulados. */
    Eigen::MatrixXd matrizDataFrame(columnas, filas);

    for (int i = 0; i < filas; ++i) {
        for (int j = 0; j < columnas; ++j) {
            matrizDataFrame(j,i) = atof(setDatos[i][j].c_str());  // Se guardan de tipo float (atof).
        }
    }

    /* Se transpone la matriz ara obtener filas por columnas*/
    return matrizDataFrame.transpose();
}

/* Para desarrolar el algoritmo de machine learning, el cual sera regresion lineal por minimos cuadrados
 * ordinarios, se usaran los datos del dataset (winedata.csv) el cual se realizara para multiples variables.
 * Dada la naturaleza de RL, si se tiene valores con diferentes unidades (ordenes de magnitud), una variable
 * podria beneficiar/estropear otra(s) variable(s): Se necesitara estandarizar los datos, dando a todas
 * las variables el mismo orden de magnitud y centradas en 0. Para ellos contruiremos una funcion de
 * normalizacion basada en el Z-Score. Se necesitan entonces 3 funciones: promedio, desviacion estandar y
 * la normalizacion Z-Score. */

/* La palabra clave auto especifica que el tipo de la variable que se empieza a declarar se deducira automaticamente
 * de su inicializador y, para las funciones, si su tipo de retorno es auto, se evaluara mediante la expresion del
 * tipo de retorno en tiempo de ejecucion. */

/*auto extoeigen::Promedio(Eigen::MatrixXd datos){
    return datos.colwise().mean();
}*/

/* En C++ la herencia del tipo de dato no es directa o no se sabe que tipo de dato debe retornar,
 * entonces se declara el tipo en una expresion "decltype" con el fin de tener seguridad de que tipo
 * de dato retornara la funcion. */

auto extoeigen::Promedio(Eigen::MatrixXd datos) -> decltype (datos.colwise().mean()){
    return datos.colwise().mean();
}

/* Para implementar la funcion de desviacion estandar los datos seran xi - Promedio y de esa forma
 * obtener la desviacion estandar. */

auto extoeigen::DesviacionEstandar(Eigen::MatrixXd data) -> decltype(((data.array().square().colwise().sum())/(data.rows()-1)).sqrt()){
    return ((data.array().square().colwise().sum())/(data.rows()-1)).sqrt();
}

/* Acto seguido se necesita aplizar el promedio y la desviacion estandar a los datos para hacer la
 * normalizacion (Z-Score) */
Eigen::MatrixXd extoeigen::Normalizacion(Eigen::MatrixXd datos) {

    Eigen::MatrixXd datos_escalados = datos.rowwise() - Promedio(datos);

    Eigen::MatrixXd matrixNorm = datos_escalados.array().rowwise()/DesviacionEstandar(datos_escalados);

    return matrixNorm;
}

/* Se implementa la funcion para dividir el conjunto de datos en entrenamiento y prueba, en el
 * dataSet se observan 12 columnas o variables. Las 11 primeras columnas corresponden a las
 * variables independientes identificadas en la literatura como "FEATURES". La ultima columna
 * (12va columna) corresponde a la variable dependiente conocida en la literatura como
 * "TARGET".
 *
 * La funcion debe retornar 4 conjuntos de datos, a saber:
 * X_train: Conjunto de datos de entrenamiento de las FEATURES.
 * y_train: Conjunto de datos de entrenamiento de la TARGET.
 * x_test: Conjunto de datos de prueba de las FEATURES.
 * y_test: Conjunto de datos de prueba de la TARGET. */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> extoeigen::TrainTestSplit(Eigen::MatrixXd datos, float sizeTrain){
    int filas = datos.rows();
    int filasEntrenamiento = round(filas * sizeTrain);
    int filasPrueba = filas - filasEntrenamiento;

    /* Con Eigen se puede especificar un bloque de una matriz: Por ejemplo, se pueden seleccionar
     * las filas superiores para el conjunto de entrenamiento indicando cuantas filas se desean,
     * seleccionando desde 0. */
    Eigen::MatrixXd entrenamiento = datos.topRows(filasEntrenamiento);

    /* Una vez seleccionadas las filas superiores, se seleccionan las columnas de la izquierda,
     * correspondientes a las FEATURES o variables independientes. */
    Eigen::MatrixXd X_train = entrenamiento.leftCols(datos.cols() -1);

    /* Seleccionamos la variable dependiente o TARGET, que corresponde a la ultima columna */
    Eigen::MatrixXd y_train = entrenamiento.rightCols(1);

    /* Seguidamente se repite el procedimiento para el conjunto de pruebas */
    Eigen::MatrixXd pruebas = datos.bottomRows(filasPrueba);
    Eigen::MatrixXd x_test = pruebas.leftCols(datos.cols() - 1);
    Eigen::MatrixXd y_test = pruebas.rightCols(1);

    /* Al retornar la tupla se empaqueta (make_tuple) para enviarse como objeto */
    return std::make_tuple(X_train, y_train, x_test, y_test);
}

/* Se crean dos funciones para exportar los valores */
void extoeigen::VectorToFile(std::vector<float> vector, std::string nameFile){
    std::ofstream ficheroSalida(nameFile);
    std::ostream_iterator<float> iteradorSalida(ficheroSalida, "\n");
    std::copy(vector.begin(), vector.end(), iteradorSalida);
}

void extoeigen::EigenToFile(Eigen::MatrixXd datos, std::string nameFile){
    std::ofstream ficheroSalida(nameFile);
    if(ficheroSalida.is_open()){
        ficheroSalida << datos << "\n";
    }
}
