#pragma once

#include <Eigen/Dense>
#include <iostream>

inline Eigen::VectorXd HostSolveLDLT(const Eigen::MatrixXd& A,
                                     const Eigen::VectorXd& b)
{
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    if (ldlt.info() != Eigen::Success)
    {
        // Fallback a une factorisation QR pour éviter de crash
        return Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(A).solve(b);
    }
    return ldlt.solve(b);
}

inline Eigen::VectorXd HostSolveQR(const Eigen::MatrixXd& A,
                                   const Eigen::VectorXd& b)
{
    const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
    if (qr.info() != Eigen::Success)
    {
        // Pas de fallback étant donné que la QR est deja notre solution rapide
        // la plus polyvalente
        throw std::runtime_error("Unable to perform QR decomposition on A!");
    }
    return qr.solve(b);
}
