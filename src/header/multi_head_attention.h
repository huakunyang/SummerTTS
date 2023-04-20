#ifndef _MULTI_HEAD_ATTENTION_H_
#define _MULTI_HEAD_ATTENTION_H_

#include "stdint.h"
#include <vector>
#include <Eigen/Dense>

using Eigen::MatrixXf;

class multi_head_attention
{
public:
    multi_head_attention(float * modelData, int32_t & offset);
    ~multi_head_attention();
    MatrixXf forward(MatrixXf x, MatrixXf c);

private:
    MatrixXf attention(const MatrixXf & query, const MatrixXf & key, const MatrixXf & value);
    MatrixXf get_relative_embeddings(const MatrixXf & relativeEmbeddings, int32_t length);
    std::vector<MatrixXf> relative_position_to_absolute_position(const std::vector<MatrixXf> & x);
    std::vector<MatrixXf> absolute_position_to_relative_position(const std::vector<MatrixXf> & x);
    void * priv_;

};

#endif
