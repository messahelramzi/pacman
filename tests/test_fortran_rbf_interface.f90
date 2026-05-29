!
! This file is subject to the terms and conditions defined in
! file 'LICENSE', which is part of this source code package.
!

!> @file test_fortran_rbf_interface.f90
!> @brief Fortran equivalent of test_cpp_rbf_interface.cpp — uses pacman_mod.
!>
!> Generates a 3-D regular-grid point cloud identical to the coarse mesh
!> used by the C++ and Python RBF tests (20 × 20 × 20 = 8 000 points,
!> spacing ≈ 0.053), evaluates the 3-D Franke reference function, then calls
!> pacman_rbf_interpolate for every Wendland basis (C0 / C2 / C4 / C6 / C8)
!> and validates the relative L2 error against the same tolerance used in
!> test_cpp_rbf_interface.cpp and test_rbf-pum.py.
!>
!> Mesh correspondence:
!>   "coarse" ~ 0.03.npz  (spacing ≈ 0.053, 20 × 20 × 20 = 8 000 pts)
!>
!> Test scenario (same as C++ test):
!>   same : source = coarse, target = coarse, tol = 1.0e-8
!>
!> Exit codes:
!>   0  — all assertions passed
!>   1  — at least one relative L2 error exceeded the tolerance
!>   77 — no Kokkos execution space is available (CTest skip)

program test_fortran_rbf_interface
  use iso_c_binding
  use pacman_mod
  implicit none

  ! ------------------------------------------------------------------
  ! Grid parameters (must match test_cpp_rbf_interface.cpp: N = 20)
  ! ------------------------------------------------------------------
  integer, parameter :: N    = 20
  integer, parameter :: NPTS = N * N * N   ! 8 000
  integer, parameter :: DIM  = 3

  ! ------------------------------------------------------------------
  ! Point cloud arrays
  ! Layout: pts(DIM, NPTS)  — Fortran column-major = C row-major
  ! ------------------------------------------------------------------
  real(c_double), allocatable :: pts(:,:)   ! (DIM, NPTS)
  real(c_double), allocatable :: vals(:)    ! (NPTS)
  real(c_double), allocatable :: out(:)     ! (NPTS)

  ! ------------------------------------------------------------------
  ! Test bookkeeping
  ! ------------------------------------------------------------------
  real(c_double), parameter :: TOL_SAME = 1.0d-8

  integer(c_int) :: execSpace
  character(len=16) :: execSpaceName

  integer :: i, j, k, idx
  real(c_double) :: step, x, y, z
  real(c_double) :: num, den, rel_l2
  integer(c_int) :: ierr
  logical :: ok, all_pass
  integer :: ret

  ! RBF function table
  integer, parameter :: NRBF = 5
  integer(c_int) :: rbfIds(NRBF)
  character(len=12) :: rbfNames(NRBF)

  ! ------------------------------------------------------------------
  ! Select execution space (best available, same priority as C++ test)
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

  write(*,'(A,A,A)') 'test_fortran_rbf_interface [', trim(execSpaceName), ']'

  ! ------------------------------------------------------------------
  ! Initialize Kokkos
  ! ------------------------------------------------------------------
  call pacman_kokkos_initialize()

  ! ------------------------------------------------------------------
  ! Build 20 × 20 × 20 regular grid in [0,1]^3 and evaluate Franke 3D
  ! ------------------------------------------------------------------
  allocate(pts(DIM, NPTS))
  allocate(vals(NPTS))
  allocate(out(NPTS))

  step = 1.0d0 / real(N - 1, c_double)
  idx  = 1
  do i = 0, N-1
    x = real(i, c_double) * step
    do j = 0, N-1
      y = real(j, c_double) * step
      do k = 0, N-1
        z = real(k, c_double) * step
        pts(1, idx) = x
        pts(2, idx) = y
        pts(3, idx) = z
        vals(idx)   = franke_3d(x, y, z)
        idx = idx + 1
      end do
    end do
  end do

  ! ------------------------------------------------------------------
  ! RBF function table
  ! ------------------------------------------------------------------
  rbfIds(1) = PACMAN_WENDLANDC0;  rbfNames(1) = 'WENDLANDC0'
  rbfIds(2) = PACMAN_WENDLANDC2;  rbfNames(2) = 'WENDLANDC2'
  rbfIds(3) = PACMAN_WENDLANDC4;  rbfNames(3) = 'WENDLANDC4'
  rbfIds(4) = PACMAN_WENDLANDC6;  rbfNames(4) = 'WENDLANDC6'
  rbfIds(5) = PACMAN_WENDLANDC8;  rbfNames(5) = 'WENDLANDC8'

  ! ------------------------------------------------------------------
  ! Run test: same mesh (source = target = coarse)
  ! ------------------------------------------------------------------
  all_pass = .true.

  do i = 1, NRBF
    out(:) = 0.0d0

    call pacman_rbf_interpolate(DIM, execSpace, rbfIds(i), &
                                pts, vals, pts, out, ierr)

    if (ierr /= 0) then
      write(*,'(A,A,A,I0)') '  ERROR  same  ', trim(rbfNames(i)), &
                             '  ierr=', int(ierr)
      all_pass = .false.
      cycle
    end if

    ! Relative L2 error: ||out - vals||_2 / ||vals||_2
    num = 0.0d0
    den = 0.0d0
    do j = 1, NPTS
      num = num + (out(j) - vals(j)) ** 2
      den = den + vals(j) ** 2
    end do
    if (den > 0.0d0) then
      rel_l2 = sqrt(num / den)
    else
      rel_l2 = sqrt(num)
    end if

    ok = (rel_l2 < TOL_SAME)
    if (ok) then
      write(*,'(A,A,A,ES12.4,A,ES8.1)') &
        '  PASS  same  ', trim(rbfNames(i)), &
        '  rel_l2=', rel_l2, '  tol=', TOL_SAME
    else
      write(*,'(A,A,A,A,ES12.4,A,ES8.1)') &
        '  FAIL  same  ', trim(rbfNames(i)), &
        '  rel_l2=', rel_l2, '  tol=', TOL_SAME
      all_pass = .false.
    end if
  end do

  ! ------------------------------------------------------------------
  ! Finalize Kokkos and report
  ! ------------------------------------------------------------------
  call pacman_kokkos_finalize()

  deallocate(pts, vals, out)

  if (all_pass) then
    ret = 0
  else
    ret = 1
  end if
  stop ret

contains

  ! ----------------------------------------------------------------
  ! Franke 3-D reference function (port of franke_functions.py:franke_3d
  ! and test_cpp_rbf_interface.cpp:franke_3d).
  ! ----------------------------------------------------------------
  pure real(c_double) function franke_3d(x, y, z)
    real(c_double), intent(in) :: x, y, z
    franke_3d =                                                        &
        0.75d0 * exp(-( (9.0d0*x-2.0d0)**2                             &
                      + (9.0d0*y-2.0d0)**2                             &
                      + (9.0d0*z-2.0d0)**2 ) / 4.0d0)                  &
      + 0.75d0 * exp(-( (9.0d0*x+1.0d0)**2 / 49.0d0                    &
                      + (9.0d0*y+1.0d0)    / 10.0d0                    &
                      + (9.0d0*z+1.0d0)    / 10.0d0 ))                 &
      + 0.50d0 * exp(-( (9.0d0*x-7.0d0)**2                             &
                      + (9.0d0*y-3.0d0)**2                             &
                      + (9.0d0*z-5.0d0)**2 ) / 4.0d0)                  &
      - 0.20d0 * exp(-( (9.0d0*x-4.0d0)**2                             &
                      + (9.0d0*y-7.0d0)**2                             &
                      + (9.0d0*z-5.0d0)**2 ))
  end function franke_3d

end program test_fortran_rbf_interface
