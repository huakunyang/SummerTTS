#ifndef _FFN_H_
#define _FFN_H_

#include <Eigen/Dense>
using Eigen::MatrixXf;

class FFN
{
public:
    FFN(float * modelData, int32_t & offset);
    ~FFN();
    MatrixXf forward(const MatrixXf & x);
private:
    void * priv_;
};

#endif
