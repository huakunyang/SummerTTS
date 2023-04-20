#include "nn_gelu.h"
#include "nn_tanh.h"

#define CONST1 (0.7978845608028654)
#define CONST2 (0.044715)

MatrixXf nn_gelu(const MatrixXf & x)
{
    MatrixXf tmp1 = ((x.array() + x.array().pow(3)*CONST2)*CONST1).matrix();
    MatrixXf tmp2 = nn_tanh(tmp1);
    tmp1 = (((tmp2.array()+1.0).array()) * x.array() * 0.5).matrix();

    return tmp1;
}
