!
! This file is subject to the terms and conditions defined in
! file 'LICENSE', which is part of this source code package.
!

!> @file test_fortran_fe_interface.f90
!> @brief Fortran equivalent of test_cpp_fe_interface.cpp — uses pacman_mod.
!>
!> Builds the same small two-cell 2-D mesh (VTK_QUAD + VTK_TRIANGLE) used by
!> test_cpp_fe_interface.cpp and test_pybindings.py, sets source values equal
!> to the x-coordinate of each node, and validates that every FE method
!> supported by the C++ test (INTERP_CLAMP) reproduces the expected 
!> x-coordinates at
!> the target points within tolerance 1e-8.
!>
!> Mesh layout (see test_cpp_fe_interface.cpp for ASCII art):
!>
!>       5
!>      /\
!>     /  \
!>  4 /____\ 3
!>   |      |
!>   |______|
!>  1        2
!>
!>  Cell 1: VTK_QUAD (9)  — vertices 1,2,3,4
!>  Cell 2: VTK_TRI  (5)  — vertices 4,3,5
!>
!> Source values = x-coordinate of each node (sp_values = sp(:,1)).
!> Reference     = x-coordinate of each target node (tp_ref = tp(:,1)).
!>
!> Exit codes:
!>   0  — all assertions passed
!>   1  — at least one value exceeded tolerance
!>   77 — no Kokkos execution space available (CTest skip)

program test_fortran_fe_interface
  use iso_c_binding
  use pacman_mod
  implicit none

  ! ------------------------------------------------------------------
  ! Mesh constants
  ! ------------------------------------------------------------------
  integer, parameter :: DIM    = 2
  integer, parameter :: NSP    = 5    ! source nodes
  integer, parameter :: NTP    = 11   ! target nodes
  integer, parameter :: NCONN  = 7    ! connectivity values
  integer, parameter :: NOFF   = 3    ! connectivity offsets (nElems+1)
  integer, parameter :: NELEM  = 2    ! cell count

  ! ------------------------------------------------------------------
  ! Source mesh
  ! Layout: pts(DIM, nPoints)  — Fortran column-major = C row-major
  ! ------------------------------------------------------------------
  real(c_double), parameter :: sp(DIM, NSP) = reshape( &
      [0.0d0, 0.0d0, &   ! node 1
       1.0d0, 0.0d0, &   ! node 2
       1.0d0, 1.0d0, &   ! node 3
       0.0d0, 1.0d0, &   ! node 4
       0.5d0, 1.5d0], &  ! node 5
      shape=[DIM, NSP])

  real(c_double), parameter :: spValues(NSP) = &
      [0.0d0, 1.0d0, 1.0d0, 0.0d0, 0.5d0]  ! x-coordinates

  ! ------------------------------------------------------------------
  ! Connectivity (CSR, VTK ordering) : F-style 1-based indexing 
  ! ------------------------------------------------------------------
  integer(c_int), parameter :: connVal(NCONN) = &
      [1, 2, 3, 4,  4, 3, 5]   ! cell0: quad 1-2-3-4 | cell1: tri 4-3-5
  integer(c_int), parameter :: connOff(NOFF)  = [0, 4, 7]

  ! ------------------------------------------------------------------
  ! Target mesh (11 nodes)
  ! ------------------------------------------------------------------
  real(c_double), parameter :: tp(DIM, NTP) = reshape( &
      [0.00d0, 0.00d0, &   ! node  1
       1.00d0, 0.00d0, &   ! node  2
       1.00d0, 1.00d0, &   ! node  3
       0.00d0, 1.00d0, &   ! node  4
       0.50d0, 1.50d0, &   ! node  5
       0.50d0, 0.00d0, &   ! node  6
       1.00d0, 0.50d0, &   ! node  7
       0.50d0, 1.00d0, &   ! node  8
       0.00d0, 0.50d0, &   ! node  9
       0.25d0, 1.25d0, &   ! node 10
       0.75d0, 1.25d0], &  ! node 11
      shape=[DIM, NTP])

  ! Reference = x-coordinate of each target node
  real(c_double), parameter :: tpRef(NTP) = &
      [0.0d0, 1.0d0, 1.0d0, 0.0d0, 0.5d0, 0.5d0, 1.0d0, 0.5d0, 0.0d0, 0.25d0, 0.75d0]

  real(c_double), parameter :: TOL = 1.0d-8

  ! ------------------------------------------------------------------
  ! VTK cell types → PACMAN cell types
  ! ------------------------------------------------------------------
  integer(c_int) :: vtkTypes(NELEM)
  integer(c_int) :: cellTypes(NELEM)

  ! ------------------------------------------------------------------
  ! Working arrays
  ! ------------------------------------------------------------------
  real(c_double) :: tpVals(NTP)
  integer(c_int) :: tpStatus(NTP)

  ! ------------------------------------------------------------------
  ! Bookkeeping
  ! ------------------------------------------------------------------
  integer(c_int) :: execSpace
  character(len=16) :: execSpaceName

  ! FE method table
  integer, parameter :: NMETHODS = 1
  integer(c_int) :: methIds(NMETHODS)
  character(len=22) :: methNames(NMETHODS)

  integer :: im, i
  real(c_double) :: diff, maxdiff
  integer(c_int) :: ierr
  logical :: ok, all_pass
  integer :: ret

  ! ------------------------------------------------------------------
  ! Select execution space
  ! ------------------------------------------------------------------
  execSpace = pacman_best_execspace()
  if (execSpace < 0) then
    write(*,'(A)') '[SKIP] No Kokkos execution space available.'
    stop 77
  end if

  select case (int(execSpace))
    case (0);  execSpaceName = 'SERIAL'
    case (1);  execSpaceName = 'OPENMP'
    case (2);  execSpaceName = 'THREADS'
    case (3);  execSpaceName = 'CUDA'
    case (4);  execSpaceName = 'HIP'
    case (5);  execSpaceName = 'SYCL'
    case default; execSpaceName = 'UNKNOWN'
  end select

  write(*,'(A,A,A)') 'test_fortran_fe_interface [', trim(execSpaceName), ']'

  ! ------------------------------------------------------------------
  ! Initialize Kokkos
  ! ------------------------------------------------------------------
  call pacman_kokkos_initialize()

  ! ------------------------------------------------------------------
  ! Convert VTK type IDs → PACMAN cell-type codes
  !   9 = VTK_QUAD,  5 = VTK_TRIANGLE
  ! ------------------------------------------------------------------
  vtkTypes(1) = 9
  vtkTypes(2) = 5
  call pacman_vtk_to_pacman_cell_type(vtkTypes, cellTypes, ierr)
  if (ierr /= 0) then
    write(*,'(A,I0)') 'ERROR vtk_to_pacman_cell_type: ierr=', int(ierr)
    call pacman_kokkos_finalize()
    stop 1
  end if

  ! ------------------------------------------------------------------
  ! FE method table (mirrors test_cpp_fe_interface.cpp + test_fe.py)
  ! ------------------------------------------------------------------
  methIds(1) = PACMAN_FE_INTERP_CLAMP;    methNames(1) = 'INTERP_CLAMP'

  ! ------------------------------------------------------------------
  ! Run all FE methods
  ! ------------------------------------------------------------------
  all_pass = .true.

  do im = 1, NMETHODS
    tpVals(:)  = 0.0d0
    tpStatus(:) = 0

    call pacman_fe_interpolate(DIM, execSpace, methIds(im),    &
                               sp, spValues,                   &
                               connVal, connOff, cellTypes,    &
                               tp, tpVals, tpStatus, ierr)

    if (ierr /= 0) then
      write(*,'(A,A,A,I0)') '  ERROR  ', trim(methNames(im)), '  ierr=', int(ierr)
      all_pass = .false.
      cycle
    end if

    ! Max absolute error against reference x-coordinates
    maxdiff = 0.0d0
    do i = 1, NTP
      diff    = abs(tpVals(i) - tpRef(i))
      maxdiff = max(maxdiff, diff)
    end do

    ok = (maxdiff <= TOL)
    if (ok) then
      write(*,'(A,A,A,ES10.3,A,ES8.1)') &
        '  PASS  ', trim(methNames(im)), &
        '  max_abs=', maxdiff, '  tol=', TOL
    else
      write(*,'(A,A,A,ES10.3,A,ES8.1)') &
        '  FAIL  ', trim(methNames(im)), &
        '  max_abs=', maxdiff, '  tol=', TOL
      all_pass = .false.
    end if
  end do

  ! ------------------------------------------------------------------
  ! Finalize and report
  ! ------------------------------------------------------------------
  call pacman_kokkos_finalize()

  if (all_pass) then
    ret = 0
  else
    ret = 1
  end if
  stop ret

end program test_fortran_fe_interface
