#include <gtest/gtest.h>
#include <cmath>
#include <fstream>

#include "reo.h"
#include "structures.h"

#define PI 3.14159625

template<typename T>
void expectNearVec(T v1, T v2)
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

    Eigen::Vector3d lc_edge{1.0, 1.0, 1.0};
    std::vector<Eigen::Vector3d> lc_edges{lc_edge};

    REO optimizer{edges, lcs, edge_covars, lc_covars, lc_edges};

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

    Eigen::Vector3d lc_edge{1.0, 1.0, 1.0};
    std::vector<Eigen::Vector3d> lc_edges{lc_edge};

    REO optimizer{edges, lcs, edge_covars, lc_covars, lc_edges};

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

TEST(Transform, AskedToInvertTransform_ReturnsCorrectInversion)
{
    Eigen::Vector3d transform(1.5, -2.0, PI/2.0);
    Eigen::Vector3d inv_transform{reo_structs::invertTransform(transform)};

    Eigen::Vector3d true_inverse{2.0, 1.5, -PI/2.0};

    expectNearVec(true_inverse, inv_transform);
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

        Eigen::Vector3d lc_edge1{0.0, 0.0, .78539};
        Eigen::Vector3d lc_edge2{0.0, 0.0, -.78539};
        Eigen::Vector3d lc_edge3{1.0, 1.0, -2.35619};
        Eigen::Vector3d lc_edge4{1.0, 1.0, .78539};
        Eigen::Vector3d lc_edge5{.5, 1.5, -2.35619};
        std::vector<Eigen::Vector3d> lc_edges{lc_edge1, lc_edge2, lc_edge3, lc_edge4, lc_edge5};
        m_lc_edges = lc_edges;

        Eigen::Vector3d covar{1e-4, 1e-4, 1e-2};
        for(int i{0}; i < edges.size(); i++)
            m_edge_covars.push_back(covar);

        for(int i{0}; i < lcs.size(); i++)
            m_lc_covars.push_back(covar);
    }
};

TEST_F(HouseREO, AskedForAnEdgeResidual_ReturnsCorrectResidual)
{
    double Tx{1.0};
    double Ty{0.0};
    double Tphi{PI/2.0};

    Eigen::Vector3d covar{1e3, 1e3,1e2};

    reo_structs::EdgeResidual res(Tx, Ty, Tphi, covar);
    double zx{this->m_edges[1][0]};
    double zy{this->m_edges[1][1]};
    double phi{this->m_edges[1][2]};
    double* residuals{new double[3]};

    res(&zx, &zy, &phi, residuals);
    double* truth{new double[3]};
    truth[0] = -.36461;
    truth[1] = 0.0;
    truth[2] = .4143;

    expectNearVec(truth, residuals);

    delete[] truth;
    delete[] residuals;
}

TEST_F(HouseREO, AskedForALCResidual_ReturnsCorrectResidual)
{
    double Tx{-1.0};
    double Ty{1.0};
    double Tphi{-3.0*PI/4.0};

    Eigen::Vector3d covar{1e3, 1e3,1e2};

    int from_id{7};
    int to_id{1};

    reo_structs::LCResidual res(Tx, Ty, Tphi, covar, from_id - to_id);
    double* residuals{new double[3]};

    double** parameters{new double*[3 * (from_id - to_id)]};

    for(int i{to_id}; i < from_id; i++)
    {
        Eigen::Vector3d edge{this->m_edges[i]};
        parameters[3*(i - to_id)] = new double[1];
        parameters[3*(i - to_id)][0] = edge[0];

        parameters[3*(i - to_id)+1] = new double[1];
        parameters[3*(i - to_id)+1][0] = edge[1];

        parameters[3*(i - to_id)+2] = new double[1];
        parameters[3*(i - to_id)+2][0] = edge[2];
    }

    res(parameters, residuals);
    double* truth{new double[3]};
    truth[0] = -60.3427;
    truth[1] = 6.1112;
    truth[2] = 2.0464;

    expectNearVec(truth, residuals);
}

TEST_F(HouseREO, AskedIfProblemIsSetUpCorrectly_ReturnsCorrectNumberOfResidualBlocksAndParameters)
{
    this->setUpOptimization();

    int num_residual_blocks{this->m_problem.NumResidualBlocks()};
    int num_parameters{this->m_problem.NumParameters()};

    int true_num_res_blocks{13};
    int true_num_parameters{24};

    EXPECT_EQ(true_num_parameters, num_parameters);
    EXPECT_EQ(true_num_res_blocks, num_residual_blocks);
}

TEST_F(HouseREO, AskedForOptimizedEdges_ReturnsCorrectWithinTolerance)
{
    this->setUpOptimization();
    std::vector<Eigen::Vector3d> opt_edges{this->solveOptimization()};

    Eigen::Vector3d edge1{1.0, 0.0, 1.570798};
    Eigen::Vector3d edge2{1.0, 0.0, 1.570798};
    Eigen::Vector3d edge3{1.0, 0.0, 1.570798};
    Eigen::Vector3d edge4{1.0, 0.0, 2.356194};
    Eigen::Vector3d edge5{sqrt(2.0), 0.0, 1.570798};
    Eigen::Vector3d edge6{sqrt(2.0)/2.0, 0.0, 1.570798};
    Eigen::Vector3d edge7{sqrt(2.0)/2.0, 0.0, 1.570798};
    Eigen::Vector3d edge8{sqrt(2.0), 0.0, 0.0};
    std::vector<Eigen::Vector3d> true_edges{edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8};

    for(int i{0}; i<true_edges.size(); i++)
    {
        for(int j{0}; j< 2; j++)
            EXPECT_NEAR(true_edges[i][j], opt_edges[i][j], .075);
        EXPECT_NEAR(true_edges[i][2], opt_edges[i][2], .12);
    }
}

//TEST(Filename, AskedToReadInFileToOptimize_ReadsCorrectValues)
//{
//    std::string filename{"../../../final-project-b4sgren/libs/reo_optimization/test_file.txt"};
//    REO optimizer(filename);

//    std::vector<Eigen::Vector3d> edges{optimizer.getEdges()};
//    std::vector<Eigen::Vector3d> edge_covars{optimizer.getEdgeCovar()};
//    std::vector<Eigen::Vector3d> lc_edges{optimizer.getLCEdges()};
//    std::vector<Eigen::Vector3d> lc_covars{optimizer.getLCCovars()};
//    std::vector<Eigen::Vector2i> lcs{optimizer.getLCS()};

//    Eigen::Vector3d edge{1.0, 0.0, 1.5708};
//    Eigen::Vector3d covar{.001, .001, .1};
//    Eigen::Vector3d lc_edge{.025, .13, .25};
//    Eigen::Vector2i lc{4, 0};

//    std::vector<Eigen::Vector3d> t_edges{edge, edge, edge, edge};
//    std::vector<Eigen::Vector3d> t_edge_covars{covar, covar, covar, covar};
//    std::vector<Eigen::Vector3d> t_lc_edges{lc_edge};
//    std::vector<Eigen::Vector3d> t_lc_covars{covar};
//    std::vector<Eigen::Vector2i> t_lcs{lc};

//    for(int i{0}; i<t_edges.size(); i++)
//    {
//        EXPECT_EQ(t_edges[i], edges[i]);
//        EXPECT_EQ(t_edge_covars[i], edge_covars[i]);
//    }

//    for(int i{0}; i < t_lcs.size(); i++)
//    {
//        EXPECT_EQ(t_lc_edges[i], lc_edges[i]);
//        EXPECT_EQ(t_lc_covars[i], lc_covars[i]);
//        EXPECT_EQ(t_lcs[i], lcs[i]);
//    }
//}
