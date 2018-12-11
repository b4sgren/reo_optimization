#ifndef REO_H
#define REO_H

#include <Eigen/Dense>
#include <vector>
#include <ceres/ceres.h>
#include "structures.h"

typedef ceres::DynamicAutoDiffCostFunction<reo_structs::LCResidual> LC_CostFunction;
typedef ceres::AutoDiffCostFunction<reo_structs::EdgeResidual, 3, 1, 1, 1> Odom_CostFunction;

class REO
{
public:
    REO();
    REO(std::vector<Eigen::Vector3d> edges, std::vector<Eigen::Vector2i> lcs,
        std::vector<Eigen::Vector3d> edge_covars, std::vector<Eigen::Vector3d> lc_covars,
        std::vector<Eigen::Vector3d> lc_edges);

    bool canSolve();
    void setUpOptimization();
    std::vector<Eigen::Vector3d> solveOptimization();

protected:
    void setUpOdometry();
    void setUpLoopClosures();
    void setUpOptions();

    std::vector<double*> setLCParameters(int from_id, int to_id, LC_CostFunction* cost_function);

    std::vector<Eigen::Vector3d> m_edges;
    std::vector<Eigen::Vector3d> m_edge_covars;

    std::vector<Eigen::Vector2i> m_lcs;
    std::vector<Eigen::Vector3d> m_lc_covars;
    std::vector<Eigen::Vector3d> m_lc_edges;

    ceres::Problem m_problem;
    ceres::Solver::Options m_options;
};

#endif
