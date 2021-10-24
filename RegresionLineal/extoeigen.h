#ifndef EXTOEIGEN_H
#define EXTOEIGEN_H

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>

class extoeigen
{
    /* Se crean los 3 atributos como argumentos de entrada a nuestro constructor
     * Nombre del dataset: */
    std::string setDatos;

    /* Separador de columnas: */
    std::string delimitador;

    /* Si tiene o no cabeza: */
    bool header;
public:
    extoeigen(std::string datos, std::string separador, bool head):
        setDatos(datos),
        delimitador(separador),
        header(head){}

    std::vector<std::vector<std::string>> LeerCSV();
    Eigen::MatrixXd CSVtoEigen(std::vector<std::vector<std::string>> setDatos, int filas, int columnas);
    auto Promedio(Eigen::MatrixXd datos) -> decltype(datos.colwise().mean());
    auto DesviacionEstandar(Eigen::MatrixXd datos) -> decltype(((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt());
    Eigen::MatrixXd Normalizacion(Eigen::MatrixXd datos);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd datos, float sizeTrain);
    void VectorToFile(std::vector<float> vector, std::string nameFile);
    void EigenToFile(Eigen::MatrixXd datos, std::string nameFile);
};

#endif // EXTOEIGEN_H
