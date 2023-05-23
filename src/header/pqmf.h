#ifndef _H_PQMF_H_
#define _H_PQMF_H_

#include <Eigen/Dense>
using Eigen::MatrixXf;

class pqmf
{
public:
    pqmf(int32_t subBands);
    ~pqmf();
    MatrixXf forward(const MatrixXf & inputMat);

private:
    void * priv_;

};

#endif
