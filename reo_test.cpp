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

//class HouseREO: public REO, public ::testing::Test
//{
//public:
//    HouseREO()
//};
