!
! This file is subject to the terms and conditions defined in
! file 'LICENSE', which is part of this source code package.
!

!> @brief Fortran module providing a high-level interface to the PACMAN
!>        interpolation library via ISO C bindings.
!>
!> Usage example (RBF-PUM):
!> @code{.f90}
!>   use pacman_mod
!>   real(8), allocatable :: srcPts(:,:), srcVals(:), tgtPts(:,:), tgtVals(:)
!>   integer :: ierr
!>
!>   call pacman_kokkos_initialize()
!>   allocate(tgtVals(nTarget))
!>
!>   call pacman_rbf_interpolate(3, PACMAN_OPENMP, PACMAN_WENDLANDC2, &
!>                               srcPts, srcVals, tgtPts, tgtVals, ierr)
!>
!>   call pacman_kokkos_finalize()
!> @endcode
!>
!> Point arrays must be laid out as @c pts(spaceDimension, nPoints), which
!> in Fortran column-major storage is identical to C row-major
!> @c pts[nPoints][spaceDimension].
module pacman_mod
  use iso_c_binding
  implicit none
  private

  ! -----------------------------------------------------------------
  ! Public procedures
  ! -----------------------------------------------------------------
  public :: pacman_kokkos_initialize
  public :: pacman_kokkos_finalize
  public :: pacman_best_execspace
  public :: pacman_rbf_interpolate
  public :: pacman_fe_interpolate
  public :: pacman_vtk_to_pacman_cell_type
  public :: pacman_vtk_cell_dim

  ! -----------------------------------------------------------------
  ! Execution space constants (match PACMAN::ExecSpaces)
  ! -----------------------------------------------------------------
  integer(c_int), parameter, public :: PACMAN_SERIAL  = 0
  integer(c_int), parameter, public :: PACMAN_OPENMP  = 1
  integer(c_int), parameter, public :: PACMAN_THREADS = 2
  integer(c_int), parameter, public :: PACMAN_CUDA    = 3
  integer(c_int), parameter, public :: PACMAN_HIP     = 4
  integer(c_int), parameter, public :: PACMAN_SYCL    = 5

  ! -----------------------------------------------------------------
  ! RBF function constants (match PACMAN::RbfFunctions)
  !   0x10 = 16, 0x11 = 17, 0x12 = 18, 0x13 = 19, 0x14 = 20
  ! -----------------------------------------------------------------
  integer(c_int), parameter, public :: PACMAN_WENDLANDC0 = 16
  integer(c_int), parameter, public :: PACMAN_WENDLANDC2 = 17
  integer(c_int), parameter, public :: PACMAN_WENDLANDC4 = 18
  integer(c_int), parameter, public :: PACMAN_WENDLANDC6 = 19
  integer(c_int), parameter, public :: PACMAN_WENDLANDC8 = 20

  ! -----------------------------------------------------------------
  ! FE method constants (match PACMAN::TransferMethods)
  !   0xF0 = 240, 0xF1 = 241, ..., 0xF4 = 244
  ! -----------------------------------------------------------------
  integer(c_int), parameter, public :: PACMAN_FE_NEAREST_NEAREST = 240
  integer(c_int), parameter, public :: PACMAN_FE_INTERP_CLAMP    = 241
  integer(c_int), parameter, public :: PACMAN_FE_INTERP_NEAREST  = 242
  integer(c_int), parameter, public :: PACMAN_FE_INTERP_ZEROFILL = 243
  integer(c_int), parameter, public :: PACMAN_FE_INTERP_EXTRAP   = 244

  ! -----------------------------------------------------------------
  ! ISO C interfaces to the plain-C shim (fortran_interface.cpp)
  ! -----------------------------------------------------------------
  interface
    subroutine pacman_kokkos_initialize_c() &
        bind(C, name='pacman_kokkos_initialize_c')
    end subroutine

    subroutine pacman_kokkos_finalize_c() &
        bind(C, name='pacman_kokkos_finalize_c')
    end subroutine

    function pacman_best_execspace_c() result(es) &
        bind(C, name='pacman_best_execspace_c')
      import :: c_int
      integer(c_int) :: es
    end function

    function pacman_rbf_interpolate_c(                           &
        spaceDimension, execSpace, rbfFunction,                  &
        sourcePoints, nSourcePoints, sourceValues,               &
        targetPoints, nTargetPoints, targetValues) result(ierr)  &
        bind(C, name='pacman_rbf_interpolate_c')
      import :: c_int, c_ptr
      integer(c_int), value :: spaceDimension
      integer(c_int), value :: execSpace
      integer(c_int), value :: rbfFunction
      type(c_ptr),    value :: sourcePoints
      integer(c_int), value :: nSourcePoints
      type(c_ptr),    value :: sourceValues
      type(c_ptr),    value :: targetPoints
      integer(c_int), value :: nTargetPoints
      type(c_ptr),    value :: targetValues
      integer(c_int)        :: ierr
    end function

    function pacman_fe_interpolate_c(                                       &
        spaceDimension, execSpace, method,                                  &
        sourcePoints, nSourcePoints, sourceValues,                          &
        connVal, connValSize, connOff, connOffSize, cellTypes,              &
        targetPoints, nTargetPoints, targetValues, targetStatus) result(ierr) &
        bind(C, name='pacman_fe_interpolate_c')
      import :: c_int, c_ptr
      integer(c_int), value :: spaceDimension
      integer(c_int), value :: execSpace
      integer(c_int), value :: method
      type(c_ptr),    value :: sourcePoints
      integer(c_int), value :: nSourcePoints
      type(c_ptr),    value :: sourceValues
      type(c_ptr),    value :: connVal
      integer(c_int), value :: connValSize
      type(c_ptr),    value :: connOff
      integer(c_int), value :: connOffSize
      type(c_ptr),    value :: cellTypes
      type(c_ptr),    value :: targetPoints
      integer(c_int), value :: nTargetPoints
      type(c_ptr),    value :: targetValues
      type(c_ptr),    value :: targetStatus
      integer(c_int)        :: ierr
    end function

    function pacman_vtk_to_pacman_cell_type_c(vtkTypes, pacmanTypes, n) &
        result(ierr) bind(C, name='pacman_vtk_to_pacman_cell_type_c')
      import :: c_int, c_ptr
      type(c_ptr),    value :: vtkTypes
      type(c_ptr),    value :: pacmanTypes
      integer(c_int), value :: n
      integer(c_int)        :: ierr
    end function

    function pacman_vtk_cell_dim_c(vtkCellType) result(dim) &
        bind(C, name='pacman_vtk_cell_dim_c')
      import :: c_int
      integer(c_int), value, intent(in) :: vtkCellType
      integer(c_int) :: dim
    end function
  end interface

contains

  ! -----------------------------------------------------------------
  ! Kokkos lifecycle wrappers
  ! -----------------------------------------------------------------

  !> @brief Initialize Kokkos with default settings.
  !> Must be called before any interpolation subroutine.
  subroutine pacman_kokkos_initialize()
    call pacman_kokkos_initialize_c()
  end subroutine

  !> @brief Finalize Kokkos.
  !> Must be called after all interpolation calls are complete.
  subroutine pacman_kokkos_finalize()
    call pacman_kokkos_finalize_c()
  end subroutine

  !> @brief Return the best execution space available in the current build.
  !> @return Integer constant from PACMAN_SERIAL / PACMAN_OPENMP / etc.
  function pacman_best_execspace() result(es)
    integer(c_int) :: es
    es = pacman_best_execspace_c()
  end function

  ! -----------------------------------------------------------------
  ! RBF-PUM interpolation
  ! -----------------------------------------------------------------

  !> @brief Interpolate scalar values from source to target points using
  !>        RBF-PUM (no mesh connectivity required).
  !>
  !> @param spaceDimension  Geometric dimension (1, 2, or 3).
  !> @param execSpace       Execution-space constant (e.g. PACMAN_OPENMP).
  !> @param rbfFunction     RBF basis-function constant (e.g. PACMAN_WENDLANDC2).
  !> @param sourcePoints    Source coordinates, shape (spaceDimension, nSource).
  !> @param sourceValues    Source scalar values, shape (nSource).
  !> @param targetPoints    Target coordinates, shape (spaceDimension, nTarget).
  !> @param targetValues    Interpolated output, shape (nTarget).  Must be
  !>                        allocated by the caller before the call.
  !> @param ierr            Optional error code: 0 = success.
  subroutine pacman_rbf_interpolate(spaceDimension, execSpace, rbfFunction, &
      sourcePoints, sourceValues, targetPoints, targetValues, ierr)
    integer,        intent(in)            :: spaceDimension
    integer(c_int), intent(in)            :: execSpace
    integer(c_int), intent(in)            :: rbfFunction
    real(c_double), intent(in),  contiguous, target :: sourcePoints(:,:)
    real(c_double), intent(in),  contiguous, target :: sourceValues(:)
    real(c_double), intent(in),  contiguous, target :: targetPoints(:,:)
    real(c_double), intent(out), contiguous, target :: targetValues(:)
    integer(c_int), intent(out), optional :: ierr

    integer(c_int) :: ret, nsrc, ntgt

    nsrc = int(size(sourceValues), c_int)
    ntgt = int(size(targetValues), c_int)
    ret  = pacman_rbf_interpolate_c(                                        &
               int(spaceDimension, c_int), execSpace, rbfFunction,          &
               c_loc(sourcePoints), nsrc, c_loc(sourceValues),              &
               c_loc(targetPoints), ntgt, c_loc(targetValues))
    if (present(ierr)) ierr = ret
  end subroutine

  ! -----------------------------------------------------------------
  ! FE interpolation
  ! -----------------------------------------------------------------

  !> @brief Interpolate scalar values from a source mesh to target points
  !>        using finite-element transfer methods.
  !>
  !> @param spaceDimension  Geometric dimension (1, 2, or 3).
  !> @param execSpace       Execution-space constant.
  !> @param method          FE method constant (e.g. PACMAN_FE_INTERP_CLAMP).
  !> @param sourcePoints    Source coordinates, shape (spaceDimension, nSource).
  !> @param sourceValues    Source scalar values, shape (nSource).
  !> @param connVal         CSR connectivity values, shape (connValSize).
  !> @param connOff         CSR connectivity offsets, shape (nElems + 1).
  !> @param cellTypes       PACMAN cell-type codes, shape (nElems).
  !> @param targetPoints    Target coordinates, shape (spaceDimension, nTarget).
  !> @param targetValues    Interpolated output, shape (nTarget).
  !> @param targetStatus    Transfer-status codes per target point, shape (nTarget).
  !> @param ierr            Optional error code: 0 = success.
  subroutine pacman_fe_interpolate(spaceDimension, execSpace, method,          &
      sourcePoints, sourceValues, connVal, connOff, cellTypes,                 &
      targetPoints, targetValues, targetStatus, ierr)
    integer,        intent(in)            :: spaceDimension
    integer(c_int), intent(in)            :: execSpace
    integer(c_int), intent(in)            :: method
    real(c_double), intent(in),  contiguous, target :: sourcePoints(:,:)
    real(c_double), intent(in),  contiguous, target :: sourceValues(:)
    integer(c_int), intent(in),  contiguous, target :: connVal(:)
    integer(c_int), intent(in),  contiguous, target :: connOff(:)
    integer(c_int), intent(in),  contiguous, target :: cellTypes(:)
    real(c_double), intent(in),  contiguous, target :: targetPoints(:,:)
    real(c_double), intent(out), contiguous, target :: targetValues(:)
    integer(c_int), intent(out), contiguous, target :: targetStatus(:)
    integer(c_int), intent(out), optional :: ierr

    integer(c_int) :: ret, nsrc, ntgt, nconnval, nconnoff

    nsrc     = int(size(sourceValues), c_int)
    ntgt     = int(size(targetValues), c_int)
    nconnval = int(size(connVal),      c_int)
    nconnoff = int(size(connOff),      c_int)

    ret = pacman_fe_interpolate_c(                                             &
              int(spaceDimension, c_int), execSpace, method,                   &
              c_loc(sourcePoints), nsrc,  c_loc(sourceValues),                 &
              c_loc(connVal), nconnval, c_loc(connOff), nconnoff, c_loc(cellTypes), &
              c_loc(targetPoints), ntgt, c_loc(targetValues), c_loc(targetStatus))
    if (present(ierr)) ierr = ret
  end subroutine

  ! -----------------------------------------------------------------
  ! VTK cell-type helpers
  ! -----------------------------------------------------------------

  !> @brief Convert VTK cell-type IDs to PACMAN CellType codes.
  !>
  !> @param vtkTypes    Input VTK cell-type IDs, shape (n).
  !> @param pacmanTypes Output PACMAN cell-type codes, shape (n).
  !> @param ierr        Optional error code: 0 = success.
  subroutine pacman_vtk_to_pacman_cell_type(vtkTypes, pacmanTypes, ierr)
    integer(c_int), intent(in),  contiguous, target :: vtkTypes(:)
    integer(c_int), intent(out), contiguous, target :: pacmanTypes(:)
    integer(c_int), intent(out), optional           :: ierr
    integer(c_int) :: ret, n
    n   = int(size(vtkTypes), c_int)
    ret = pacman_vtk_to_pacman_cell_type_c(c_loc(vtkTypes), c_loc(pacmanTypes), n)
    if (present(ierr)) ierr = ret
  end subroutine

  !> @brief Return the topological dimension for a VTK cell-type ID.
  !>
  !> @param vtkCellType VTK cell-type ID.
  !> @return Topological dimension (0, 1, 2, or 3); -1 on error.
  function pacman_vtk_cell_dim(vtkCellType) result(dim)
    integer(c_int), intent(in) :: vtkCellType
    integer(c_int) :: dim
    dim = pacman_vtk_cell_dim_c(vtkCellType)
  end function

end module pacman_mod
