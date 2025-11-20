#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

Eigen::MatrixXd MatrixFromCSV(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, ','))
        {
            row.push_back(std::stod(cell));
        }

        data.push_back(row);
    }

    if (data.empty())
    {
        throw std::runtime_error("File is empty or invalid: " + filename);
    }

    const size_t cols = data[0].size();
    for (const auto& row : data)
    {
        if (row.size() != cols)
        {
            throw std::runtime_error("Inconsistent number of columns in CSV "
                                     "file. The CSV file might be corrupted.");
        }
    }

    Eigen::MatrixXd mat(data.size(), cols);
    for (size_t i = 0; i < data.size(); ++i)
        for (size_t j = 0; j < cols; ++j)
            mat(i, j) = data[i][j];

    return mat;
}

Eigen::VectorXd VectorFromCSV(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<double> values;
    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ','))
        {
            if (!cell.empty())
            {
                values.push_back(std::stod(cell));
            }
        }
    }

    if (values.empty())
    {
        throw std::runtime_error("File is empty or has no numeric data: "
                                 + filename);
    }

    Eigen::VectorXd vec(values.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        vec(i) = values[i];
    }

    return vec;
}

auto VerifySolution(const Eigen::MatrixXd& A, const Eigen::VectorXd& X,
                    const Eigen::VectorXd& B)
{
    if (A.cols() != X.size())
    {
        std::cerr << "Dimension mismatch: A.cols() != X.size()" << std::endl;
        return -1.0;
    }
    if (A.rows() != B.size())
    {
        std::cerr << "Dimension mismatch: A.rows() != B.size()" << std::endl;
        return -1.0;
    }
    Eigen::VectorXd residual = A * X - B;
    return residual.norm();
}

volatile int sink;
Eigen::VectorXd SolveWithQR(const Eigen::MatrixXd& A, const Eigen::VectorXd& B,
                            std::chrono::duration<double>& time_s)
{
    const auto t1 = std::chrono::high_resolution_clock::now();
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
    auto sol = qr.solve(B);
    const auto t2 = std::chrono::high_resolution_clock::now();
    time_s = t2 - t1;

    return sol;
}

int main(int argc, char* argv[])
{
    const std::string system0 = "./systems/0";
    const std::string path_A = system0 + "/" + "A.csv";
    const std::string path_B = system0 + "/" + "B.csv";
    const std::string path_X = system0 + "/" + "X.csv";

    const Eigen::MatrixXd A = MatrixFromCSV(path_A);
    const Eigen::VectorXd B = VectorFromCSV(path_B);
    const Eigen::VectorXd X = VectorFromCSV(path_X);
    Eigen::VectorXd S;

    constexpr size_t iterations = 100;
    std::vector<double> time_ms_vec;
    time_ms_vec.reserve(iterations);

    for (size_t it = 0; it < iterations; ++it)
    {
        std::chrono::duration<double> time_s;
        S = SolveWithQR(A, B, time_s);
        time_ms_vec.push_back(time_s.count() * 1'000.0);
    }

    double avg_time_ms =
        std::accumulate(time_ms_vec.begin(), time_ms_vec.end(), 0.0)
        / time_ms_vec.size();
    std::cout << "Solving AX=B with a QR decomposition took " << std::fixed
              << std::setprecision(16) << avg_time_ms << "ms (average on "
              << iterations << " iterations)." << std::endl;

    const auto norm = VerifySolution(A, S, B);
    std::cout << "Residual norm: " << norm << std::endl;

    return 0;
}