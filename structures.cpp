#include <Eigen/Dense>

namespace reo_structs
{
double const PI{3.14159625};

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

struct EdgeResidual
{
public:
    EdgeResidual(double Tx, double Ty, double T_phi, Eigen::Vector3d co_var): m_Tx{Tx}, m_Ty{Ty}, m_Tphi{T_phi}
    {
        Eigen::Matrix3d covar{co_var.asDiagonal()};
        m_xi = covar.llt().matrixL().transpose();
    }

    Eigen::Matrix3d getXi()
    {
        return m_xi;
    }

    template<typename T>
    bool operator()(const T* const zx, const T* const zy, const T* const theta, T* residuals) const
    {
        Eigen::Matrix<T, 3, 1> T2;
        T2(0) = *zx;
        T2(1) = *zy;
        T2(2) = *theta;

        Eigen::Matrix<T, 3, 1> trans;
        trans<<T(0.0), T(0.0), T(0.0);

        trans = concatenateTransform(trans, T2);

        residuals[0] = (T(m_Tx) - T2(0)) * T(m_xi(0, 0));
        residuals[1]= (T(m_Ty) - T2(1)) * T(m_xi(1, 1));
        residuals[2] = (T(m_Tphi) - T2(2)) * T(m_xi(2, 2));
        return true;
    }

protected:
    double m_Tx, m_Ty, m_Tphi;
    Eigen::Matrix3d m_xi;
};

}
