#include "reo.h"
#include "structures.h"

#include <ceres/ceres.h>
#include <cmath>
#include <fstream>

typedef ceres::DynamicAutoDiffCostFunction<reo_structs::LCResidual> LC_CostFunction;
typedef ceres::AutoDiffCostFunction<reo_structs::EdgeResidual, 3, 1, 1, 1> Odom_CostFunction;

REO::REO(){}

REO::REO(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> edges, std::vector<Eigen::Vector2i> lcs,
         mat3d edge_covars, mat3d lc_covars,
         vec3d lc_edges)
{
    m_edges = edges;
    m_lcs = lcs;
    m_edge_covars = edge_covars;
    m_lc_covars = lc_covars;
    m_lc_edges = lc_edges;
}

REO::REO(std::string filename)
{
    readFile(filename);
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
        Eigen::Matrix3d co_var{m_edge_covars[i]};
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
        Eigen::Matrix3d co_var{m_lc_covars[i]};
        Eigen::Vector3d transform{m_lc_edges[i]};

        LC_CostFunction* cost_function{new LC_CostFunction(new reo_structs::LCResidual(transform(0), transform(1), transform(2), co_var, abs(from_id - to_id)))};
        cost_function->SetNumResiduals(3);

        std::vector<double*> parameter_blocks{setLCParameters(from_id, to_id, cost_function)};

        m_problem.AddResidualBlock(cost_function, NULL, parameter_blocks);
    }
}

void REO::readFile(std::string filename)
{
    std::ifstream fin{filename};

    if(!fin.fail())
    {
        int from_id, to_id;
        double x, y, phi, covar_x, covar_y, covar_p;

        while(fin >> from_id >> to_id >> x >> y >> phi >> covar_x >> covar_y >> covar_p)
        {
            if(to_id - from_id == 1)
            {
                m_edges.push_back(Eigen::Vector3d{x, y, phi});
                Eigen::Matrix3d temp_covar;
                temp_covar << covar_x, 0, 0, 0, covar_y, 0, 0, 0, covar_p;
                m_edge_covars.push_back(temp_covar);
            }
            else
            {
                m_lc_edges.push_back(Eigen::Vector3d{x, y, phi});
                Eigen::Matrix3d temp_covar;
                temp_covar << covar_x, 0, 0, 0, covar_y, 0, 0, 0, covar_p;
                m_lc_covars.push_back(temp_covar);
                m_lcs.push_back(Eigen::Vector2i{from_id, to_id});
            }
        }
    }
    fin.close();
}

void REO::setUpOptions()
{
    m_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    m_options.function_tolerance = 1e-15;
}

std::vector<double*> REO::setLCParameters(int from_id, int to_id, LC_CostFunction* cost_function)
{
    std::vector<double*> parameters;
    parameters.clear();

    if(from_id < to_id)
    {
        int temp = to_id;
        to_id = from_id;
        from_id = temp;
    }

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

vec3d REO::solveOptimization()
{
    ceres::Solver::Summary summary;
    ceres::Solve(m_options, &m_problem, &summary);

    vec3d opt_edges;
    opt_edges.clear();

    for(int i{0}; i < m_edges.size(); i++)
        opt_edges.push_back(m_edges[i]);

    return opt_edges;
}

vec3d REO::getEdges() const
{
    return m_edges;
}

mat3d REO::getEdgeCovar() const
{
    return m_edge_covars;
}

vec3d REO::getLCEdges() const
{
    return m_lc_edges;
}

mat3d REO::getLCCovars() const
{
    return m_lc_covars;
}

std::vector<Eigen::Vector2i> REO::getLCS() const
{
    return m_lcs;
}
