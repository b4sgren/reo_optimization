#include <gtest/gtest.h>
#include "reo.h"
#include "structures.cpp"
#include <cmath>

#define PI 3.14159625

void expectNearVec(Eigen::Vector3d v1, Eigen::Vector3d v2)
{
    for(int i{0}; i<3; i++)
        EXPECT_NEAR(v1[i], v2[i], .001);
}

TEST(VectorOfEdgesLoopClosuresAndCovariance, AskedIfInformationIsCorrect_ReturnsTrue)
{
    Eigen::Vector3d edge;
    edge << 1.0, 0.0, PI/2.0;
    std::vector<Eigen::Vector3d> edges{edge, edge, edge, edge};

    Eigen::Vector2i lc;
    lc << 3, 1;
    std::vector<Eigen::Vector2i> lcs{lc};

    Eigen::Vector3d edge_covar;
    edge_covar << 1e-5, 1e-5, 1e-3;
    std::vector<Eigen::Vector3d> edge_covars{edge_covar, edge_covar, edge_covar, edge_covar};

    Eigen::Vector3d lc_covar;
    lc_covar << 1e-3, 1e-3, 1e-1;
    std::vector<Eigen::Vector3d> lc_covars{lc_covar};

    REO optimizer = REO(edges, lcs, edge_covars, lc_covars);

    EXPECT_TRUE(optimizer.canSolve());
}

TEST(REOWithDifferentVectorLengths, AskedIfSolvable_ReturnsFalse)
{
    Eigen::Vector3d edge;
    edge << 0.0, 0.0, PI/2.0;
    std::vector<Eigen::Vector3d> edges{edge, edge, edge, edge};

    Eigen::Vector2i lc;
    lc << 2, 1;
    std::vector<Eigen::Vector2i> lcs{lc};

    Eigen::Vector3d edge_covar;
    edge_covar << 1e-3, 1e-3, 1e-1;
    std::vector<Eigen::Vector3d> edge_covars;

    Eigen::Vector3d lc_covar;
    lc_covar << 1e-2, 1e-2, 1e-2;
    std::vector<Eigen::Vector3d> lc_covars{lc_covar};

    REO optimizer = REO(edges, lcs, edge_covars, lc_covars);

    EXPECT_FALSE(optimizer.canSolve());
}

TEST(TwoEdges, AskedToConcatenate_ReturnsCorrectConcatenation)
{
    Eigen::Vector3d edge{1.0, 0.0, PI/2.0};
    Eigen::Vector3d init_x{0.0, 0.0, 0.0};

    Eigen::Vector3d pos{reo_structs::concatenateTransform(init_x, edge)};
    pos = reo_structs::concatenateTransform(pos, edge);

    Eigen::Vector3d truth{1.0, 1.0, PI};


    expectNearVec(truth, pos);
}

TEST(EdgeResidual, PassedInCovarianceMatrix_ConvertsToSquareRootOfCovariance)
{
    Eigen::Vector3d covar{1e5, 1e5, 1e3};
    double Tx{1.0};
    double Ty{1.0};
    double Tphi{1.0};

    reo_structs::EdgeResidual res(Tx, Ty, Tphi, covar);
    Eigen::Matrix3d var{res.getXi()};
    Eigen::Vector3d sqrt_covar{var(0, 0), var(1, 1), var(2, 2)};

    Eigen::Vector3d truth{sqrt(covar(0)), sqrt(covar(1)), sqrt(covar(2))};

    expectNearVec(truth, sqrt_covar);
}

class HouseREO: public REO, public ::testing::Test
{
public:
    HouseREO()
    {
        Eigen::Vector3d edge1{1.0, 0.0, 1.570798};
        Eigen::Vector3d edge2{1.01153, 0.0, 1.529368};
        Eigen::Vector3d edge3{0.98064, 0.0, 1.599686};
        Eigen::Vector3d edge4{1.03755, 0.0, 2.18385};
        Eigen::Vector3d edge5{1.41222, 0.0, 1.65611};
        Eigen::Vector3d edge6{0.67212, 0.0, 1.539428};
        Eigen::Vector3d edge7{0.69755, 0.0, 1.49709};
        Eigen::Vector3d edge8{1.45754, 0.0, 0.0};
        std::vector<Eigen::Vector3d> edges{edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8};
        m_edges = edges;

        Eigen::Vector2i lc1{0, 4};
        Eigen::Vector2i lc2{2, 5};
        Eigen::Vector2i lc3{1, 7};
        Eigen::Vector2i lc4{3, 8};
        Eigen::Vector2i lc5{0, 6};
        std::vector<Eigen::Vector2i> lcs{lc1, lc2, lc3, lc4, lc5};
        m_lcs = lcs;

        Eigen::Vector3d covar{1e-3, 1e-3, 1e-2};
        for(int i{0}; i < edges.size(); i++)
            m_edge_covars.push_back(covar);

        for(int i{0}; i < lcs.size(); i++)
            m_lc_covars.push_back(covar);
    }
};

TEST_F(HouseREO, AskedForAnEdgeResidual_ReturnsCorrectResidual)
{

}
