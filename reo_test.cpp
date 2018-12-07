#include <gtest/gtest.h>
#include "reo.h"

#define PI 3.14159625

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
