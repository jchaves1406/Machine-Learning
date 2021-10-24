#include "modellinealregression.h"

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

/* Se necesita entrenar el modelo, lo que implica minimizar la funcion de costo
 * de esta forma se puede medir la funcion de hipotesis. La funcion de costo es la forma
 * de penalizar al modelo por cometer un error */
float ModelLinealRegression::FuncionCosto(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta){
    Eigen::MatrixXd diferencia = pow((X * theta - y).array(), 2);

    return (diferencia.sum()/(2 * X.rows()));
}

/* Se necesita proveer al programa una funcion para dar al algoritmo los valores iniciales de theta,
 * el cual variará o cambiará iterativamente hasta que converja al valor minimo de nuestra funcion de costo.
 * Basicamente esto representa el gradiente descendiente, el cual es las derivadas parciales de la funcion.
 * Las entradas para la funcion serán X (FEATURES) y (TARGET), aplha(LEARNING RATE) y el numero de
 * iteraciones (numero de veces que se actualizara theta hasta que la funcion converja). */
std::tuple<Eigen::VectorXd , std::vector<float>> ModelLinealRegression::GradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha, int iteraciones){
    /* Almacenamiento temporal de thetas */
    Eigen::MatrixXd temporal = theta;

    /* Ahora necesitamos la cantidad de parametros m (FEATURES) */
    int parametros = theta.rows();

    /* Costo inicial: se actualizará con los nuevos pesos */
    std::vector<float> costo;
    costo.push_back(FuncionCosto(X, y, theta));

    /* Por cada iteracion se calcula la función de error. Se actualiza theta y se calcula el nuevo
     * valor de la funcion de costo para los nuevos valores de theta */

    for(int i = 0; i < iteraciones; i++){
        Eigen::MatrixXd error = X * theta - y;
        for(int j = 0; j < parametros; j++){
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd termino = error.cwiseProduct(X_i);
            temporal(j, 0) = theta(j, 0) - ((alpha/X.rows())*termino.sum());
        }
        theta = temporal;
        costo.push_back(FuncionCosto(X, y, theta));
    }

    return std::make_tuple(theta, costo);
}

/* Se crea la metrica r2 */
float ModelLinealRegression::R2Cuadrado(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){
    auto numerador = pow((y - y_hat).array(),2).sum();
    auto denominador = pow(y.array() - y.mean(),2).sum();
    return 1-(numerador/denominador);
}
