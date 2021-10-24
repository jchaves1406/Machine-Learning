#ifndef MODELLINEALREGRESSION_H
#define MODELLINEALREGRESSION_H

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

class ModelLinealRegression
{
public:
    ModelLinealRegression(){}

    float FuncionCosto(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta);
    std::tuple<Eigen::VectorXd , std::vector<float>> GradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha, int iteraciones);
    float R2Cuadrado(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);
};

#endif // MODELLINEALREGRESSION_H
