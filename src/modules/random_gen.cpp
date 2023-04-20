#include<random>
#include<Eigen/Eigen>
#include<ctime>
using namespace std;
using namespace Eigen;

MatrixXf rand_gen(int32_t row, int32_t col, float mean, float std)
{
    static default_random_engine e(time(0));
    static normal_distribution<float> n(mean,std);

    MatrixXf m = MatrixXf::Zero(row,col).unaryExpr([](float dummy){return n(e);});
    return m;
}

