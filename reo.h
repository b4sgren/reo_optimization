#ifndef REO_H
#define REO_H

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <vector>
#include <ceres/ceres.h>
#include "structures.h"

typedef ceres::DynamicAutoDiffCostFunction<reo_structs::LCResidual> LC_CostFunction;
typedef ceres::AutoDiffCostFunction<reo_structs::EdgeResidual, 3, 1, 1, 1> Odom_CostFunction;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vec3d;
typedef std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> mat3d;
typedef std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>> vec2i;

class REO
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    REO();
    REO(vec3d edges, vec2i lcs,
        mat3d edge_covars, mat3d lc_covars,
        vec3d lc_edges);
    REO(std::string filename);

    bool canSolve();
    void setUpOptimization();
    vec3d solveOptimization();

    vec3d getEdges() const;
    mat3d getEdgeCovar() const;
    vec3d getLCEdges() const;
    mat3d getLCCovars() const;
    vec2i getLCS() const;

protected:
    void setUpOdometry();
    void setUpLoopClosures();
    void setUpOptions();
    void readFile(std::string filename);

    std::vector<double*> setLCParameters(int from_id, int to_id, LC_CostFunction* cost_function);

    vec3d m_edges;
    mat3d m_edge_covars;

    vec2i m_lcs;
    mat3d m_lc_covars;
    vec3d m_lc_edges;

    ceres::Problem m_problem;
    ceres::Solver::Options m_options;
};

#endif
