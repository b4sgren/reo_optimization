#include "structures.h"

namespace reo_structs
{
template<typename T>
Eigen::Matrix<T, 3, 1> concatenateTransform(Eigen::Matrix<T, 3, 1> T1, Eigen::Matrix<T, 3, 1> T2)
{
    T cs = cos(T1(2));
    T ss = sin(T1(2));

    T x = T1(0) + T2(0) * cs - T2(1) * ss;
    T y = T1(1) + T2(0) * ss + T2(1) * cs;
    T psi = T1(2) + T2(2);

    while(psi > T(PI))
        psi -= T(2*PI);
    while(psi < T(-PI))
        psi += T(2*PI);

    return Eigen::Matrix<T, 3, 1>(x, y, psi);
}

}
