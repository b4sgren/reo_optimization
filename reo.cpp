#include "reo.h"

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
