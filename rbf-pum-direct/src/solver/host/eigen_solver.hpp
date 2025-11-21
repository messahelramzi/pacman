#pragma once

#include <Eigen/Dense>

inline Eigen::VectorXd HostSolveLDLT(const Eigen::MatrixXd& A,
                                     const Eigen::VectorXd& b)
{
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    if (ldlt.info() != Eigen::Success)
    {
        throw std::runtime_error("Unable to perform LDLT decomposition on A!");
    }
    return ldlt.solve(b);
}

inline Eigen::VectorXd HostSolveQR(const Eigen::MatrixXd& A,
                                   const Eigen::VectorXd& b)
{
    const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
    if (qr.info() != Eigen::Success)
    {
        throw std::runtime_error("Unable to perform QR decomposition on A!");
    }
    return qr.solve(b);
}
