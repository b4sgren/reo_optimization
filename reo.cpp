#include "reo.h"
#include "structures.h"
#include <ceres/ceres.h>

typedef ceres::DynamicAutoDiffCostFunction<reo_structs::LCResidual> LC_CostFunction;
typedef ceres::AutoDiffCostFunction<reo_structs::EdgeResidual, 3, 1, 1, 1> Odom_CostFunction;

REO::REO(){}

REO::REO(std::vector<Eigen::Vector3d> edges, std::vector<Eigen::Vector2i> lcs,
         std::vector<Eigen::Vector3d> edge_covars, std::vector<Eigen::Vector3d> lc_covars)
{
    m_edges = edges;
    m_lcs = lcs;
    m_edge_covars = edge_covars;
    m_lc_covars = lc_covars;
}

bool REO::canSolve()
{
    if(m_edges.size() == 0 || m_edge_covars.size() == 0
            || m_lcs.size() == 0 || m_lc_covars.size() == 0)
        return false;
    else if(m_edges.size() == m_edge_covars.size()
            && m_lcs.size() == m_lc_covars.size())
        return true;
    else
        return false;
}

void REO::setUpOptimization()
{
    setUpOdometry();
    setUpLoopClosures();
    setUpOptions();
}

void REO::setUpOdometry()
{
    for(int i{0}; i<m_edges.size(); i++)
    {
        Eigen::Vector3d co_var{m_edge_covars[i]};
        Eigen::Vector3d transform{m_edges[i]};

        Odom_CostFunction* cost_function{new Odom_CostFunction{new reo_structs::EdgeResidual(transform(0), transform(1), transform(2), co_var)}};
        m_problem.AddResidualBlock(cost_function, NULL, &m_edges[i](0), &m_edges[i](1), &m_edges[i](2));
    }
}

void REO::setUpLoopClosures()
{
    for(int i{0}; i < m_lcs.size(); i++)
    {
        int from_id{m_lcs[i](0)};
        int to_id{m_lcs[i](1)};
        Eigen::Vector3d co_var{m_lc_covars[i]};
        Eigen::Vector3d transform{getLCTransform(from_id, to_id)};

        LC_CostFunction* cost_function{new LC_CostFunction(new reo_structs::LCResidual(transform(0), transform(1), transform(2), co_var, from_id - to_id))};
        cost_function->SetNumResiduals(3);

        std::vector<double*> parameter_blocks{setLCParameters(from_id, to_id, cost_function)};

        m_problem.AddResidualBlock(cost_function, NULL, parameter_blocks);
    }
}

void REO::setUpOptions()
{
    m_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    m_options.function_tolerance = 1e-15;
}

Eigen::Vector3d REO::getLCTransform(int from_id, int to_id)
{
    Eigen::Vector3d transform{0.0, 0.0, 0.0};
    for(int i{to_id}; i < from_id; i++)
    {
        transform = reo_structs::concatenateTransform(transform, m_edges[i]);
    }

    return transform;
}

std::vector<double*> REO::setLCParameters(int from_id, int to_id, LC_CostFunction* cost_function)
{
    std::vector<double*> parameters;
    parameters.clear();

    for(int i{to_id}; i < from_id; i++)
    {
        parameters.push_back(&m_edges[i][0]);
        cost_function->AddParameterBlock(1);
        parameters.push_back(&m_edges[i][1]);
        cost_function->AddParameterBlock(1);
        parameters.push_back(&m_edges[i][2]);
        cost_function->AddParameterBlock(1);
    }

    return parameters;
}

std::vector<Eigen::Vector3d> REO::solveOptimization()
{
    ceres::Solver::Summary summary;
    ceres::Solve(m_options, &m_problem, &summary);

    std::vector<Eigen::Vector3d> opt_edges;
    opt_edges.clear();

    for(int i{0}; i < m_edges.size(); i++)
        opt_edges.push_back(m_edges[i]);

    return opt_edges;
}
