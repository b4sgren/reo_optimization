#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <Eigen/Dense>

namespace reo_structs
{
double const PI{3.14159625};

template<typename T>
Eigen::Matrix<T, 3, 1> invertTransform(Eigen::Matrix<T, 3, 1>& T1)
{
    T cs{cos(T1[2])};
    T ss{sin(T1[2])};

    T dx{-(T1[0] * cs + T1[1] * ss)};
    T dy{-(-T1[0] * ss + T1[1] * cs)};

    return Eigen::Matrix<T, 3, 1>{dx, dy, -T1[2]};
}

template<typename T>
Eigen::Matrix<T, 3, 1> concatenateTransform(Eigen::Matrix<T, 3, 1>& T1, Eigen::Matrix<T, 3, 1>& T2)
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
    EdgeResidual(double Tx, double Ty, double T_phi, Eigen::Matrix3d& co_var): m_Tx{Tx}, m_Ty{Ty}, m_Tphi{T_phi}
    {
        m_xi = co_var.llt().matrixL().transpose();
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

        Eigen::Matrix<T, 3, 3> xi;
        xi << T(m_xi(0, 0)), T(m_xi(0, 1)), T(m_xi(0, 2)), T(m_xi(1, 0)), T(m_xi(1, 1)), T(m_xi(1, 2)), T(m_xi(2, 0)), T(m_xi(2, 1)), T(m_xi(2, 2));
        Eigen::Matrix<T, 3, 1> T0;
        T0 << T(m_Tx), T(m_Ty), T(m_Tphi);

        Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);
        res = xi * (T0 - trans);
        return true;
    }

protected:
    double m_Tx, m_Ty, m_Tphi;
    Eigen::Matrix3d m_xi;
};


struct LCResidual
{
public:
    LCResidual(double Tx, double Ty, double T_phi, Eigen::Matrix3d& co_var, int edge_count): m_Tx{Tx}, m_Ty{Ty}, m_Tphi{T_phi}, m_num_edges{edge_count}
    {
        m_xi = co_var.llt().matrixL().transpose();
    }

    template<typename T>
    bool operator()(T const* const* parameters, T* residuals) const
    {
        Eigen::Matrix<T, 3, 1> trans;
        trans << T(0.0), T(0.0), T(0.0);
        for(int i{0}; i < m_num_edges; i++)
        {
            T zx = parameters[3 * i][0];
            T zy = parameters[3* i + 1][0];
            T theta = parameters[3 * i + 2][0];
            Eigen::Matrix<T, 3, 1> temp;
            temp << zx, zy, theta;
            trans = concatenateTransform(trans, temp);
        }
        trans = invertTransform(trans);

        Eigen::Matrix<T, 3, 3> xi;
        xi << T(m_xi(0, 0)), T(m_xi(0, 1)), T(m_xi(0, 2)), T(m_xi(1, 0)), T(m_xi(1, 1)), T(m_xi(1, 2)), T(m_xi(2, 0)), T(m_xi(2, 1)), T(m_xi(2, 2));
        Eigen::Matrix<T, 3, 1> T0;
        T0 << T(m_Tx), T(m_Ty), T(m_Tphi);

        Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);
        res = xi * (T0 - trans);
        return true;
    }

protected:
    double m_Tx, m_Ty, m_Tphi;
    int m_num_edges;
    Eigen::Matrix3d m_xi;
};

}

#endif
