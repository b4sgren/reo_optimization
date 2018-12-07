#ifndef REO_H
#define REO_H

#include <Eigen/Dense>
#include <vector>


class REO
{
public:
    REO(std::vector<Eigen::Vector3d> edges, std::vector<Eigen::Vector2i> lcs,
        std::vector<Eigen::Vector3d> edge_covars, std::vector<Eigen::Vector3d> lc_covars);

    bool canSolve();

protected:
    std::vector<Eigen::Vector3d> m_edges;
    std::vector<Eigen::Vector3d> m_edge_covars;

    std::vector<Eigen::Vector2i> m_lcs;
    std::vector<Eigen::Vector3d> m_lc_covars;
};

#endif
