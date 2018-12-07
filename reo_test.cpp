#include <gtest/gtest.h>
#include "reo.h"

#define PI 3.14159625

TEST(VectorOfEdgesLoopClosuresAndCovariance, AskedIfInformationIsCorrect_ReturnsYes)
{
    Eigen::Vector3d edge;
    edge << 1.0, 0.0, PI/2.0;
    std::vector<Eigen::Vector3d> edges{edge, edge, edge, edge};

    Eigen::Vector2i lc;
    lcs << 3, 1;
    std::vector<Eigen::Vector2i> lcs{lc};

    Eigen::Vector3d edge_covar;
    edge_covar << 1e-5, 1e-5, 1e-3;
    std::vector<Eigen::Vector3d> edge_covars{edge_covar, edge_covar, edge_covar, edge_covar};

    Eigen::Vector3d lc_covar;
    lc_covar << 1e-3, 1e-3, 1e-1;
    std::vector<Eigen::Vector3d> lc_covars{lc_covar};

    REO optimizer{edges, lcs, edge_covars, lc_covars};

    EXPECT_TRUE(optimizer.canSolve());
}
