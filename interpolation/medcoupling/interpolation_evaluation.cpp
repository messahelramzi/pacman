#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#include "med.h"
#include "MEDCoupling.hxx"
#include "MEDCouplingField.hxx"
#include "MEDCouplingFieldDouble.hxx"
#include "MEDCouplingFieldTemplate.hxx"
#include "MEDCouplingMemArray.hxx"
#include "MEDCouplingMesh.hxx"
#include "MEDCouplingRemapper.hxx"
#include "MEDFileMesh.hxx"

using namespace MEDCoupling;

static MEDCouplingMesh *get_mesh(const std::string &path)
{
    const MEDFileMesh *file_mesh = MEDFileMesh::New(path.c_str());
    MEDCouplingMesh *mesh = file_mesh->getMeshAtLevel(0);
    file_mesh->decrRef();
    return mesh;
}

static inline void set_field_info(MEDCouplingField *field, const MEDCouplingMesh *mesh, const std::string &name, const NatureOfField nature)
{
    field->setMesh(mesh);
    field->setName(name);
    field->setNature(nature);
}

static TypeOfField get_type_of_field(const std::string &interpolation_mode, bool first_call = true)
{
    if (!interpolation_mode.compare("P0P0"))
    {
        return ON_CELLS;
    }
    if (!interpolation_mode.compare("P1P1"))
    {
        return ON_NODES;
    }
    if (!interpolation_mode.compare("P0P1"))
    {
        if (first_call)
        {
            return ON_CELLS;
        }
        return ON_NODES;
    }
    if (!interpolation_mode.compare("P1P0"))
    {
        if (first_call)
        {
            return ON_NODES;
        }
        return ON_CELLS;
    }
    throw new std::invalid_argument("Invalid interpolation mode!");
}

static std::pair<MEDCouplingFieldDouble *, std::chrono::duration<double, std::nano>> eval_interp(const std::string &source_mesh_path, const std::string &target_mesh_path, const std::string &interp_mode, const NatureOfField nature, const std::string &output_file)
{
    const std::string franke_function = "0.75*exp(-(((9.*x-2.)*(9.*x-2.))+((9.*y-2.)*(9.*y-2.))+((9.*z-2.)*(9.*z-2.)))/4.)+0.75*exp(-(((9.*x+1.)*(9.*x+1.))/49.+(9.*y+1.)/10.+(9.*z+1.)/10.))+0.5*exp(-(((9.*x-7.)*(9.*x-7.))+((9.*y-3.)*(9.*y-3.))+((9.*z-5.)*(9.*z-5.))/4.))-0.2*exp(-(((9.*x-4.)*(9.*x-4.))+((9.*y-7.)*(9.*y-7.))+((9.*z-5.)*(9.*z-5.))))";

    MCAuto<MEDCouplingMesh> source_mesh = get_mesh(source_mesh_path);
    MCAuto<MEDCouplingFieldDouble> source_data_field = source_mesh->fillFromAnalytic(get_type_of_field(interp_mode), 1, franke_function);
    set_field_info(source_data_field, source_mesh, "FrankeData", nature);

    MCAuto<MEDCouplingMesh> target_mesh = get_mesh(target_mesh_path);
    MEDCouplingRemapper remapper = MEDCouplingRemapper();
    auto t1 = std::chrono::high_resolution_clock::now().time_since_epoch();
    remapper.prepare(source_mesh, target_mesh, interp_mode);
    auto t2 = std::chrono::high_resolution_clock::now().time_since_epoch();

    MCAuto<MEDCouplingFieldDouble> target_data_field = remapper.transferField(source_data_field, 1e300);
    set_field_info(target_data_field, target_mesh, "InterpolatedData", nature);

    // std::cout << "Interpolation Matrix Computation Time: " << ((t2 - t1) / 1000000.0).count() << "ms" << std::endl;

    MCAuto<MEDCouplingFieldDouble> ref_data_field = target_mesh->fillFromAnalytic(get_type_of_field(interp_mode, false), 1, franke_function);
    set_field_info(ref_data_field, target_mesh, "RefFrankeData", nature);

    MCAuto<MEDCouplingFieldDouble> error_data_field = MEDCouplingFieldDouble::SubstractFields(ref_data_field, target_data_field);
    set_field_info(error_data_field, target_mesh, "ErrorData", nature);

    const std::string output_path = output_file + '_' + interp_mode;
    error_data_field->writeVTK(output_path);

    return std::pair<MEDCouplingFieldDouble *, std::chrono::duration<double, std::nano>>(error_data_field->deepCopy(), (t2 - t1) / 1000000.0);
}

static double get_average_time(double times[], int n, int k)
{
    k = k % n;
    double sum = 0.0;
    for (k; k < n * 10; k += n)
    {
        sum += times[k];
    }
    return sum / 10.0;
}

int main(void)
{
    const std::pair<std::string, std::string> meshes_pairs[4] = {
        std::pair<std::string, std::string>("0.03", "0.003"),
        std::pair<std::string, std::string>("0.02", "0.002"),
        std::pair<std::string, std::string>("0.01", "0.001"),
        std::pair<std::string, std::string>("0.004", "0.0005"),
    };

    const std::string interpolation_modes[] = {"P0P0", "P0P1", "P1P0", "P1P1"};
    const std::string output_file = "../plot_results/data";
    const NatureOfField nature = IntensiveMaximum;

    const std::string output_folder = "output_files";
    if (mkdir(output_folder.c_str(), 0777))
    {
        if (errno != EEXIST)
        {
            std::string err = strerror(errno);
            std::cerr << "Unable to create the output dir!" << std::endl;
            std::cerr << err << std::endl;
            exit(1);
        }
    }

    const std::string output_path = output_folder + '/' + output_file;

    for (unsigned short a = 0; a < 4; a++)
    {
        const std::string source_mesh_path = "meshes/" + meshes_pairs[a].first + ".med";
        const std::string target_mesh_path = "meshes/" + meshes_pairs[a].second + ".med";

        int n = sizeof(interpolation_modes) / sizeof(interpolation_modes[0]);
        MEDCouplingFieldDouble *fields[n];
        double time_ms[n * 10];
        unsigned short i = 0;
        unsigned short t_i = 0;

        for (auto interp_mode : interpolation_modes)
        {
            auto ret_pair = eval_interp(source_mesh_path, target_mesh_path, interp_mode, nature, output_path);
            auto f = ret_pair.first;
            auto t = ret_pair.second;
            fields[i++] = f;
            time_ms[t_i++] = t.count();
        }
        for (int m = 0; m < 9; m++)
        {
            for (auto interp_mode : interpolation_modes)
            {
                auto ret_pair = eval_interp(source_mesh_path, target_mesh_path, interp_mode, nature, output_path);
                auto t = ret_pair.second;
                time_ms[t_i++] = t.count();
            }
        }

        for (int o = 0; o < i; o++)
        {
            auto field = fields[o];
            DataArrayDouble *data = field->getArray()->deepCopy();
            double *data_ptr = data->getPointer();

            size_t len = data->getNbOfElems();
            size_t new_len = len;
            std::multiset<double> abs_sorted_data;

            for (size_t i = 0; i < len; i++)
            {
                if (data_ptr[i] == -1e300)
                {
                    /* Some cells of the target field cannot be matched to a cell or a node in source field by medcoupling
                    ** In this case, I chose to not consider these values when it comes to statistics
                    */
                    new_len--;
                }
                else
                {
                    abs_sorted_data.emplace(std::abs(data_ptr[i]));
                }
            }

            long k = 0;
            double sum = 0;
            double med, p90, p95, p99;
            double min, max;
            min = *(abs_sorted_data.begin());
            max = *(--(abs_sorted_data.end()));
            for (auto p = abs_sorted_data.begin(); p != abs_sorted_data.end(); p++, k++)
            {
                sum += *p;
                if (k == (long)std::floor(new_len * 0.50))
                {
                    med = *p;
                }
                if (k == (long)std::floor(new_len * 0.90))
                {
                    p90 = *p;
                }
                if (k == (long)std::floor(new_len * 0.95))
                {
                    p95 = *p;
                }
                if (k == (long)std::floor(new_len * 0.99))
                {
                    p99 = *p;
                }
            }
            double avg = sum / k;

            const std::string report_path = "result_" + interpolation_modes[o] + "_pair" + (char)(a + '0') + ".yaml";
            std::fstream report_file;
            report_file.open(output_folder + '/' + report_path, std::ios::out);
            if (!report_file.is_open())
            {
                std::string err = strerror(errno);
                std::cerr << "File failed to be opened (or created)" << std::endl;
                std::cerr << err << std::endl;
                exit(1);
            }
            report_file << std::setprecision(18);
            report_file << std::fixed;

            report_file << "values:" << std::endl;
            report_file << "    avg: " << avg << std::endl;
            report_file << "    min: " << min << std::endl;
            report_file << "    max: " << max << std::endl;
            report_file << "    med: " << med << std::endl;
            report_file << "    p90: " << p90 << std::endl;
            report_file << "    p95: " << p95 << std::endl;
            report_file << "    p99: " << p99 << std::endl;
            report_file << "    time-ms: " << get_average_time(time_ms, n, o) << std::endl;
            report_file << "    invalid-values: " << std::boolalpha << (bool)(k != len) << std::endl;
            report_file.flush();
            report_file.close();

            field->decrRef();
            data->decrRef();
        }
    }

    return 0;
}
