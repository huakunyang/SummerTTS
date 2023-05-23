#ifndef _H_ISTFT_H_
#define _H_ISTFT_H_

#include "stdint.h"
#include <Eigen/Dense>

using Eigen::MatrixXf;

class iStft
{
public:
    iStft(int32_t filterLen, int32_t hopLen, int32_t winLen);
    ~iStft();

    MatrixXf forward(const MatrixXf & mag, const MatrixXf & phase);

private:
    void * priv_;

};

#endif
