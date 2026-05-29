//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include <Kokkos_Core.hpp>
#include <stdexcept>

#include "common/concepts.hpp"
#include "common/transfer.hxx"
#include "common/types.hpp"

namespace PACMAN {
namespace FiniteElements {

/**
 * @brief Small helper to canonicalize face-node ordering for row comparison.
 * @tparam T 1D Kokkos subview type containing one face row.
 * @tparam Dim Spatial dimension controlling expected face arity.
 */
template <KokkosViewRank<1> T, int_t Dim> struct permute {
  /// @brief In-place permutation used to keep equivalent faces sortable.
  /// @param[in,out] kokkos_sv Face row subview to reorder.
  KOKKOS_INLINE_FUNCTION
  static void permuteCol(T kokkos_sv) {
    if constexpr (Dim >= 2) {
      if (kokkos_sv(0) > kokkos_sv(1)) {
        Kokkos::kokkos_swap(kokkos_sv(0), kokkos_sv(1));
      }
    }

    if constexpr (Dim >= 3) {
      if (kokkos_sv(2) > kokkos_sv(3)) {
        Kokkos::kokkos_swap(kokkos_sv(2), kokkos_sv(3));
      }
      if (kokkos_sv(0) > kokkos_sv(2)) {
        Kokkos::kokkos_swap(kokkos_sv(0), kokkos_sv(2));
      }
      if (kokkos_sv(1) > kokkos_sv(3)) {
        Kokkos::kokkos_swap(kokkos_sv(1), kokkos_sv(3));
      }
      if (kokkos_sv(1) > kokkos_sv(2)) {
        Kokkos::kokkos_swap(kokkos_sv(1), kokkos_sv(2));
      }
    }
  }
};

/**
 * @brief Build tetrahedralized boundary skin from source elements.
 *
 * The routine enumerates candidate element faces, removes interior duplicates,
 * triangulates surviving boundary faces, then stores the resulting skin faces
 * and parent element ids in the transfer object.
 *
 * @tparam ExecSpace Kokkos execution space used for scans, sorting, and
 * compaction.
 * @tparam Dim Spatial dimension of the problem (`1`, `2`, or `3`).
 * @param[in,out] transfer Transfer descriptor containing source connectivity
 * and metadata.
 * Reads: `sourcePoints`, `connValues`, `connOffsets`, `cellTypes`,
 * `nbLinearSkinFaces`.
 * Writes: `skinFaces`, `skinParents`.
 */
template <typename ExecSpace, int_t Dim>
void ComputeLinearSkin(Transfer<ExecSpace, Dim> &transfer) {
  Kokkos::Profiling::pushRegion("Compute Linear Skin");

  using MemorySpace = typename ExecSpace::memory_space;
  ExecSpace execSpace{};

  // TODO: Rewrite -> Ugly !
  auto LFPE_val_vec = getLinearFacesPaddedEntries(Dim);
  auto LFPE_val = Kokkos::View<shortint_t *,
                               Kokkos::DefaultHostExecutionSpace::memory_space,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
      LFPE_val_vec.data(), LFPE_val_vec.size());
  auto linearFacesPadded_val =
      Kokkos::create_mirror_view_and_copy(execSpace, LFPE_val);
  auto LFPE_off_vec = getLinearFacesPaddedOffsets(Dim);
  auto LFPE_off =
      Kokkos::View<offset_t *, Kokkos::DefaultHostExecutionSpace::memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
          LFPE_off_vec.data(), LFPE_off_vec.size());
  auto linearFacesPadded_off =
      Kokkos::create_mirror_view_and_copy(execSpace, LFPE_off);

  auto TF_val_vec = getTriFacesEntries(Dim);
  auto TF_val = Kokkos::View<shortint_t *,
                             Kokkos::DefaultHostExecutionSpace::memory_space,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
      TF_val_vec.data(), TF_val_vec.size());
  auto TriFaces_val = Kokkos::create_mirror_view_and_copy(execSpace, TF_val);
  auto TF_off_vec = getTriFacesOffsets(Dim);
  auto TF_off =
      Kokkos::View<offset_t *, Kokkos::DefaultHostExecutionSpace::memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(TF_off_vec.data(),
                                                            TF_off_vec.size());
  auto TriFaces_off = Kokkos::create_mirror_view_and_copy(execSpace, TF_off);

  auto TLF_val_vec = getTriLocFacesEntries(Dim);
  auto TLF_val = Kokkos::View<shortint_t *,
                              Kokkos::DefaultHostExecutionSpace::memory_space,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
      TLF_val_vec.data(), TLF_val_vec.size());
  auto TriLocFaces_val =
      Kokkos::create_mirror_view_and_copy(execSpace, TLF_val);
  auto TLF_off_vec = getTriLocFacesOffsets(Dim);
  auto TLF_off =
      Kokkos::View<offset_t *, Kokkos::DefaultHostExecutionSpace::memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(TLF_off_vec.data(),
                                                            TLF_off_vec.size());
  auto TriLocFaces_off =
      Kokkos::create_mirror_view_and_copy(execSpace, TLF_off);

  auto nbNode = transfer.sourcePoints.extent_int(0);
  auto nbLinFaces = transfer.nbLinearSkinFaces;
  auto CellTypesPtr = transfer.cellTypes;
  auto connValPtr = transfer.connValues;
  auto connOffPtr = transfer.connOffsets;

  auto nbElem = CellTypesPtr.extent(0);

  int_t nodesPerFace = -1;
  switch (Dim) {
  case 0:
    nodesPerFace = 0;
    break;
  case 1:
    nodesPerFace = 1;
    break;
  case 2:
    nodesPerFace = 2;
    break;
  case 3:
    nodesPerFace = 4;
    break;
  default:
    std::cerr << "Invalid Dim " << Dim << std::endl;
    assert(false);
    break;
  }

  auto linFaces = Kokkos::View<int_t **, ExecSpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing, "linFaces"),
      nbLinFaces, nodesPerFace + 1); // last column is CellType
  auto allFaceId = Kokkos::View<int_t *, ExecSpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing, "allFaceId"),
      nbLinFaces); // global linface id and local element face id
  auto allFaceLocId = Kokkos::View<int_t *, ExecSpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "allFaceLocId"),
      nbLinFaces); // global linface id and local element face id
  auto allParents = Kokkos::View<int_t *, ExecSpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing, "allParents"),
      nbLinFaces);

  constexpr int_t maxi = std::numeric_limits<int_t>::max();

  Kokkos::Profiling::pushRegion("Fill all faces and sort per row permute");
  // Avoid branching but very ugly: to change !
  Kokkos::parallel_scan(
      "Fill all faces and sort per row : Col permutation",
      Kokkos::RangePolicy(execSpace, 0, nbElem),
      KOKKOS_LAMBDA(const index_t &i, index_t &partial_sum, bool is_final) {
        auto ielType = static_cast<cell_t>(CellTypesPtr(i));
        auto facesPadded = Kokkos::subview(
            linearFacesPadded_val,
            Kokkos::make_pair(linearFacesPadded_off(ielType),
                              linearFacesPadded_off(ielType + 1)));
        auto localnbFaces = facesPadded.extent_int(0) / nodesPerFace;
        auto firstIndex = partial_sum;
        partial_sum += localnbFaces;
        if (is_final) {
          auto localConnectivity = Kokkos::subview(
              connValPtr, Kokkos::make_pair(connOffPtr(i), connOffPtr(i + 1)));
          for (index_t j = 0; j < localnbFaces; j++) { // loop on element faces
            auto linFaces_sv =
                Kokkos::subview(linFaces, firstIndex + j, Kokkos::ALL());
            allParents(firstIndex + j) = i;
            allFaceId(firstIndex + j) = firstIndex + j;
            allFaceLocId(firstIndex + j) = j;
            for (index_t k = 0; k < nodesPerFace - 1;
                 k++) { // loop on non-padded face connectivity
              auto index = facesPadded(j * nodesPerFace + k);
              linFaces_sv(k) = localConnectivity(index);
            }
            auto index = facesPadded((j + 1) * nodesPerFace - 1);
            linFaces_sv(nodesPerFace - 1) =
                (index == -1) ? maxi : localConnectivity(index);
            using subviewType = decltype(linFaces_sv);
            permute<subviewType, Dim>::permuteCol(linFaces_sv);
            linFaces_sv(nodesPerFace) = ielType;
          }
        }
      });
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion(
      "Create permutation vector to sort linFaces by row");
  auto firstNodeIds = Kokkos::subview(linFaces, Kokkos::ALL(), 0);

  using IdViewType = decltype(firstNodeIds);
  using CompType = Kokkos::BinOp1D<IdViewType>;
  using DeviceType = Kokkos::Device<ExecSpace, MemorySpace>;

  Kokkos::BinSort<IdViewType, CompType, DeviceType, int_t> bin_sort(
      firstNodeIds, CompType(nbLinFaces, 0, nbNode), true /*sort within*/);
  bin_sort.create_permute_vector();
  auto permutationVector = bin_sort.get_permute_vector();
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("Permute and Compute Nbtris");

  index_t nbTris = 0;
  Kokkos::parallel_reduce(
      "Permute and Compute NbTris",
      Kokkos::RangePolicy(execSpace, 0, nbLinFaces),
      KOKKOS_LAMBDA(const index_t &i, index_t &sum) {
        auto index = permutationVector(i);

        // vector who would be at pos "i" in the (first column)-sorted matrix
        // faces
        auto vect = Kokkos::subview(linFaces, index, Kokkos::ALL());

        index_t curr_row_occur =
            1; // Number of occurrences of the row (excluding current)
        index_t curr = i;
        // Find the first row starting with vect(0)
        while (curr > 0 && linFaces(permutationVector(curr), 0) == vect(0)) {
          --curr;
        }
        // In case curr is 0, we do not increase
        if (linFaces(permutationVector(curr), 0) != vect(0)) {
          ++curr;
        }
        // For all elements starting with vect(0)
        while (curr <= nbLinFaces - 1 &&
               linFaces(permutationVector(curr), 0) == vect(0)) {
          if (curr == i) { // skip current
            ++curr;
            continue;
          }
          bool diff = false;
          // Check if identical
          for (index_t t = 1; t < nodesPerFace + 1; ++t)
            if (linFaces(permutationVector(curr), t) != vect(t)) {
              diff = true;
              break;
            }
          if (!diff) {
            ++curr_row_occur;
            // if (curr_row_occur == 2)
            break;
          }
          ++curr;
        }
        if (curr_row_occur == 2) {
          allFaceId(index) = -1;
        } else {
          auto ielType = (int)CellTypesPtr(allParents(index));
          auto iTriLocFaces = Kokkos::subview(
              TriLocFaces_val, Kokkos::make_pair(TriLocFaces_off(ielType),
                                                 TriLocFaces_off(ielType + 1)));
          auto iloc = allFaceLocId(index);
          sum += (iTriLocFaces(iloc + 1) - iTriLocFaces(iloc)) / Dim;
        }
      },
      nbTris);
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("Remove");

  // remove the -1 values while saving the relative order of elements (in place)
  const auto end = Kokkos::Experimental::remove(execSpace, allFaceId, -1);
  const int dist =
      Kokkos::Experimental::distance(Kokkos::Experimental::begin(allFaceId),
                                     end); // number of not -1

  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("Finalize: compute linear skin");

  auto skinFaces = Kokkos::View<int_t **, ExecSpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing, "skinFaces"),
      nbTris, Dim);
  auto skinParents = Kokkos::View<int_t *, ExecSpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing, "skinParents"),
      nbTris);

  Kokkos::parallel_scan(
      "Fill Skin Faces & Parents", Kokkos::RangePolicy(execSpace, 0, dist),
      KOKKOS_LAMBDA(const index_t &i, index_t &partial_sum, bool is_final) {
        auto index = allFaceId(i);
        auto iParent = allParents(index);
        auto ielType = static_cast<cell_t>(CellTypesPtr(iParent));
        auto iTriFaces = Kokkos::subview(
            TriFaces_val, Kokkos::make_pair(TriFaces_off(ielType),
                                            TriFaces_off(ielType + 1)));
        auto iTriLocFaces = Kokkos::subview(
            TriLocFaces_val, Kokkos::make_pair(TriLocFaces_off(ielType),
                                               TriLocFaces_off(ielType + 1)));
        auto iloc = allFaceLocId(index);
        auto localnbFaces = (iTriLocFaces(iloc + 1) - iTriLocFaces(iloc)) / Dim;
        auto firstIndex = partial_sum;
        partial_sum += localnbFaces;
        if (is_final) {
          auto localConnectivity = Kokkos::subview(
              connValPtr,
              Kokkos::make_pair(connOffPtr(iParent), connOffPtr(iParent + 1)));
          for (index_t j = 0; j < localnbFaces; j++) {
            skinParents(firstIndex + j) = iParent;
            auto pos = iTriLocFaces(iloc) + j * Dim;
            for (int_t k = 0; k < Dim; k++) {
              const auto iconn = iTriFaces(pos + k);
              skinFaces(firstIndex + j, k) = localConnectivity(iconn);
            }
          }
        }
      });
  transfer.skinFaces = skinFaces;
  transfer.skinParents = skinParents;

  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();
}

} // namespace FiniteElements

} // namespace PACMAN
