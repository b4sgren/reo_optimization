#ifndef REO_H
#define REO_H

#include <Eigen/Dense>
#include <vector>
#include <ceres/ceres.h>
#include "structures.h"

class REO
{
public:
    REO();
    REO(std::vector<Eigen::Vector3d> edges, std::vector<Eigen::Vector2i> lcs,
        std::vector<Eigen::Vector3d> edge_covars, std::vector<Eigen::Vector3d> lc_covars);

    bool canSolve();
    void setUpOptimization();

protected:
    void setUpOdometry();
    void setUpLoopClosures();
    void setUpOptions();

    Eigen::Vector3d getLCTransform(int from_id, int to_id);
    std::vector<double*> setLCParameters(int from_id, int to_id);

    std::vector<Eigen::Vector3d> m_edges;
    std::vector<Eigen::Vector3d> m_edge_covars;

    std::vector<Eigen::Vector2i> m_lcs;
    std::vector<Eigen::Vector3d> m_lc_covars;

    ceres::Problem m_problem;
    ceres::Solver::Options m_options;
};

#endif
