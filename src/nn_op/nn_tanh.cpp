#include "nn_tanh.h"

#define BIG_NUM (1e+10)
#define SMALL_NUM (1e-8)

MatrixXf nn_tanh(const MatrixXf & x)
{
    MatrixXf x_exp = x.array().exp();
    x_exp = (x_exp.array().isInf()).select(BIG_NUM,x_exp);

    MatrixXf x_n_exp = (x.array()*(-1.0)).exp();
    x_n_exp = (x_n_exp.array().isInf()).select(BIG_NUM,x_n_exp);

    MatrixXf m0 = x_exp - x_n_exp;
    MatrixXf m1 = x_exp + x_n_exp;
    m1 = (m1.array() < SMALL_NUM).select(SMALL_NUM,m1);

    MatrixXf ret = m0.array()/m1.array();

   return ret;
}
