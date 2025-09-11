! Simple test program for SparseIR Fortran bindings
program test_ext
   use sparseir
   use sparseir_ext
   implicit none

   integer, parameter :: dp = KIND(1.0D0)

   call test_positive_only_false()
contains

   subroutine test_positive_only_true()
      type(IR) :: irobj
      double precision, parameter :: beta = 10.0_DP
      double precision, parameter :: omega_max = 2.0_DP
      double precision, parameter :: epsilon = 1.0e-10_DP
      double precision, parameter :: lambda = beta * omega_max
      logical, parameter :: positive_only = .true.

      call init_ir(irobj, beta, lambda, epsilon, positive_only)

      ! Test Fermionic case
      call test_case_target_dim_1(irobj, SPIR_STATISTICS_FERMIONIC, "Fermionic, target_dim=1, positive_only=false")
      call test_case_target_dim_2(irobj, SPIR_STATISTICS_FERMIONIC, "Fermionic, target_dim=2, positive_only=false")

      ! Test Bosonic case
      call test_case_target_dim_1(irobj, SPIR_STATISTICS_BOSONIC, "Bosonic, target_dim=1, positive_only=false")
      call test_case_target_dim_2(irobj, SPIR_STATISTICS_BOSONIC, "Bosonic, target_dim=2, positive_only=false")

      ! Finalize IR object
      call finalize_ir(irobj)
   end subroutine test_positive_only_true

   subroutine test_positive_only_false()
      type(IR) :: irobj
      double precision, parameter :: beta = 10.0_DP
      double precision, parameter :: omega_max = 2.0_DP
      double precision, parameter :: epsilon = 1.0e-10_DP
      double precision, parameter :: lambda = beta * omega_max
      logical, parameter :: positive_only = .false.

      call init_ir(irobj, beta, lambda, epsilon, positive_only)

      ! Test Fermionic case
      call test_case_target_dim_1(irobj, SPIR_STATISTICS_FERMIONIC, "Fermionic, target_dim=1, positive_only=false")
      call test_case_target_dim_2(irobj, SPIR_STATISTICS_FERMIONIC, "Fermionic, target_dim=2, positive_only=false")

      ! Test Bosonic case
      call test_case_target_dim_1(irobj, SPIR_STATISTICS_BOSONIC, "Bosonic, target_dim=1, positive_only=false")
      call test_case_target_dim_2(irobj, SPIR_STATISTICS_BOSONIC, "Bosonic, target_dim=2, positive_only=false")

      ! Finalize IR object
      call finalize_ir(irobj)
   end subroutine test_positive_only_false

   subroutine test_case_target_dim_1(obj, statistics, case_name)
      type(IR), intent(inout) :: obj
      integer, intent(in) :: statistics
      character(len=*), intent(in) :: case_name

      double precision, allocatable :: coeffs(:,:)
      complex(kind=dp), allocatable :: g_ir(:,:), g_dlr(:,:), giw(:,:), giw_reconst(:,:), g_ir2_z(:,:), gtau_z(:,:)
      integer :: i, j
      double precision :: r, r_imag
      integer :: nfreq
      logical :: positive_only
      complex(kind=dp), allocatable :: imag_tmp(:)

      integer, parameter :: target_dim = 1
      integer, parameter :: extra_dim_size = 2

      double precision, allocatable :: u_tau(:)

      print *, "Testing ", case_name

      positive_only = obj%positive_only

      if (statistics == SPIR_STATISTICS_FERMIONIC) then
         nfreq = obj%nfreq_f
      else if (statistics == SPIR_STATISTICS_BOSONIC) then
         nfreq = obj%nfreq_b
      else
         print *, "Error: Invalid statistics"
         stop
      end if

      ! Allocate arrays
      allocate(coeffs(obj%npoles, extra_dim_size))
      allocate(g_ir(obj%size, extra_dim_size))
      allocate(g_dlr(obj%npoles, extra_dim_size))
      allocate(g_ir2_z(obj%size, extra_dim_size))
      allocate(gtau_z(obj%ntau, extra_dim_size))
      allocate(giw(nfreq, extra_dim_size))
      allocate(giw_reconst(nfreq, extra_dim_size))


      ! Generate random coefficients
      do i = 1, obj%npoles
         do j = 1, extra_dim_size
            call random_number(r)
            call random_number(r_imag)
            if (positive_only) then
               coeffs(i,j) = dcmplx(2.0_DP * r - 1.0_DP, 0.0_DP) * sqrt(abs(obj%s(i)))
            else
               coeffs(i,j) = dcmplx(2.0_DP * r - 1.0_DP, r_imag) * sqrt(abs(obj%s(i)))
            end if
         end do
      end do

      ! Convert DLR coefficients to IR coefficients
      g_dlr = cmplx(coeffs, 0.0_DP, kind=DP)
      call dlr2ir(obj, target_dim, g_dlr, g_ir)

      ! Evaluate Green's function at Matsubara frequencies from IR
      call evaluate_matsubara(obj, statistics, target_dim, g_dlr, giw)

      ! Convert Matsubara frequencies back to IR
      call fit_matsubara(obj, statistics, target_dim, giw, g_ir2_z)

      ! Compare IR coefficients (using real part of g_ir2_z)
      if (.not. compare_with_relative_error_d(coeffs, real(g_ir2_z), 10.0_DP * obj%eps)) then
         print *, "Error: IR coefficients do not match after transformation cycle"
         stop
      end if

      ! Evaluate Green's function at tau points
      call evaluate_tau(obj, target_dim, g_ir2_z, gtau_z)

      ! Check tau evaluation
      allocate(u_tau(obj%size))
      u_tau = eval_u_tau(obj, obj%tau(1))
      allocate(imag_tmp(extra_dim_size))
      imag_tmp = 0.0_DP
      do i = 1, obj%size
         do j = 1, extra_dim_size
            imag_tmp(j) = imag_tmp(j) + u_tau(i) * g_ir2_z(i, j)
         end do
      end do
      if (abs(imag_tmp(1) - gtau_z(1, 1)) / max(abs(imag_tmp(1)), abs(gtau_z(1, 1))) & 
         > 10.0_DP * obj%eps) then
         print *, "Error: Tau evaluation does not match direct calculation"
         stop
      end if
      deallocate(u_tau)
      deallocate(imag_tmp)

      ! Convert tau points back to IR
      call fit_tau(obj, target_dim, gtau_z, g_ir2_z)

      ! Evaluate Green's function at Matsubara frequencies again
      call evaluate_matsubara(obj, statistics, target_dim, g_ir2_z, giw_reconst)

      ! Compare the original and reconstructed Matsubara frequencies
      if (.not. compare_with_relative_error_z(giw, giw_reconst, 10.0_DP * obj%eps)) then
         print *, "Error: Matsubara frequencies do not match after transformation cycle"
         stop
      end if

      ! Deallocate arrays
      deallocate(coeffs)
      deallocate(g_ir)
      deallocate(g_dlr)
      deallocate(g_ir2_z)
      deallocate(gtau_z)
      deallocate(giw)
      deallocate(giw_reconst)

   end subroutine test_case_target_dim_1

   subroutine test_case_target_dim_2(obj, statistics, case_name)
      type(IR), intent(inout) :: obj
      integer, intent(in) :: statistics
      character(len=*), intent(in) :: case_name

      double precision, allocatable :: coeffs(:,:)
      complex(kind=dp), allocatable :: g_ir(:,:), g_dlr(:,:), giw(:,:), giw_reconst(:,:), g_ir2_z(:,:), gtau_z(:,:)
      integer :: i, j
      double precision :: r, r_imag
      integer :: nfreq
      logical :: positive_only

      integer, parameter :: target_dim = 2
      integer, parameter :: extra_dim_size = 2

      print *, "Testing ", case_name

      positive_only = obj%positive_only

      if (statistics == SPIR_STATISTICS_FERMIONIC) then
         nfreq = obj%nfreq_f
      else if (statistics == SPIR_STATISTICS_BOSONIC) then
         nfreq = obj%nfreq_b
      else
         print *, "Error: Invalid statistics"
         stop
      end if

      ! Allocate arrays with correct dimension order for target_dim=2
      allocate(coeffs(extra_dim_size, obj%npoles))
      allocate(g_ir(extra_dim_size, obj%size))
      allocate(g_dlr(extra_dim_size, obj%npoles))
      allocate(g_ir2_z(extra_dim_size, obj%size))
      allocate(gtau_z(extra_dim_size, obj%ntau))
      allocate(giw(extra_dim_size, nfreq))
      allocate(giw_reconst(extra_dim_size, nfreq))


      ! Generate random coefficients with correct dimension order
      do j = 1, obj%npoles
         do i = 1, extra_dim_size
            call random_number(r)
            call random_number(r_imag)
            if (positive_only) then
               coeffs(i,j) = dcmplx(2.0_DP * r - 1.0_DP, 0.0_DP) * sqrt(abs(obj%s(j)))
            else
               coeffs(i,j) = dcmplx(2.0_DP * r - 1.0_DP, r_imag) * sqrt(abs(obj%s(j)))
            end if
         end do
      end do

      ! Convert DLR coefficients to IR coefficients
      g_dlr = cmplx(coeffs, 0.0_DP, kind=DP)
      call dlr2ir(obj, target_dim, g_dlr, g_ir)

      ! Evaluate Green's function at Matsubara frequencies from IR
      call evaluate_matsubara(obj, statistics, target_dim, g_dlr, giw)

      ! Convert Matsubara frequencies back to IR
      call fit_matsubara(obj, statistics, target_dim, giw, g_ir2_z)

      ! Compare IR coefficients (using real part of g_ir2_z)
      if (.not. compare_with_relative_error_d(coeffs, real(g_ir2_z), 10.0_DP * obj%eps)) then
         print *, "Error: IR coefficients do not match after transformation cycle"
         stop
      end if

      ! Evaluate Green's function at tau points
      call evaluate_tau(obj, target_dim, g_ir2_z, gtau_z)

      ! Convert tau points back to IR
      call fit_tau(obj, target_dim, gtau_z, g_ir2_z)

      ! Evaluate Green's function at Matsubara frequencies again
      call evaluate_matsubara(obj, statistics, target_dim, g_ir2_z, giw_reconst)

      ! Compare the original and reconstructed Matsubara frequencies
      if (.not. compare_with_relative_error_z(giw, giw_reconst, 10.0_DP * obj%eps)) then
         print *, "Error: Matsubara frequencies do not match after transformation cycle"
         stop
      end if

      ! Deallocate arrays
      deallocate(coeffs)
      deallocate(g_ir)
      deallocate(g_dlr)
      deallocate(g_ir2_z)
      deallocate(gtau_z)
      deallocate(giw)
      deallocate(giw_reconst)
   end subroutine test_case_target_dim_2

   function compare_with_relative_error_d(a, b, tol) result(is_close)
      double precision, intent(in) :: a(:,:), b(:,:)
      double precision, intent(in) :: tol
      logical :: is_close
      double precision :: max_diff, max_ref
      integer :: i, j

      max_diff = 0.0_DP
      max_ref = 0.0_DP

      do j = 1, size(a, 2)
         do i = 1, size(a, 1)
            max_diff = max(max_diff, abs(a(i,j) - b(i,j)))
            max_ref = max(max_ref, abs(a(i,j)))
         end do
      end do

      is_close = max_diff <= tol * max_ref

      if (.not. is_close) then
         print *, "max_diff:", max_diff
         print *, "max_ref:", max_ref
         print *, "tol:", tol
         print *, "relative error:", max_diff / max_ref
      end if
   end function compare_with_relative_error_d

   function compare_with_relative_error_z(a, b, tol) result(is_close)
      complex(kind=dp), intent(in) :: a(:,:), b(:,:)
      double precision, intent(in) :: tol
      logical :: is_close
      double precision :: max_diff, max_ref
      integer :: i, j

      max_diff = 0.0_DP
      max_ref = 0.0_DP

      do j = 1, size(a, 2)
         do i = 1, size(a, 1)
            max_diff = max(max_diff, abs(a(i,j) - b(i,j)))
            max_ref = max(max_ref, abs(a(i,j)))
         end do
      end do

      is_close = max_diff <= tol * max_ref

      if (.not. is_close) then
         print *, "max_diff:", max_diff
         print *, "max_ref:", max_ref
         print *, "tol:", tol
         print *, "relative error:", max_diff / max_ref
      end if
   end function compare_with_relative_error_z
end program test_ext
