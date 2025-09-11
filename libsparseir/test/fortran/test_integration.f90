! Simple test program for SparseIR Fortran bindings
program test_integration
   use sparseir
   use, intrinsic :: iso_c_binding
   implicit none
   real(c_double),target :: xmin, xmax, ymin, ymax
   integer(c_int), target :: status
   integer(c_int), target :: ntaus, nmatsus
   integer(c_int), target :: npoles
   integer, allocatable :: seed(:)  ! For random seed

   real(c_double), parameter :: beta = 10.0_c_double
   real(c_double), parameter :: omega_max = 2.0_c_double
   real(c_double), parameter :: epsilon = 1.0e-10_c_double
   real(c_double), parameter :: lambda = beta * omega_max

   integer(c_int), parameter :: positive_only = 0
   integer(c_int), parameter :: order = SPIR_ORDER_COLUMN_MAJOR

   ! Initialize random seed with a fixed value
   call random_seed(size=status)
   if (status > 0) then
      allocate(seed(status))
      seed = 982743  ! Fixed seed value
      call random_seed(put=seed)
      deallocate(seed)
   end if

   ! Test both Fermionic and Bosonic cases
   call test_case(SPIR_STATISTICS_FERMIONIC, "Fermionic")
   call test_case(SPIR_STATISTICS_BOSONIC, "Bosonic")

contains

   subroutine test_case(statistics, case_name)
      integer(c_int), intent(in) :: statistics
      character(len=*), intent(in) :: case_name

      type(c_ptr) :: k_ptr, k_copy_ptr
      type(c_ptr) :: sve_ptr
      type(c_ptr) :: basis_ptr, dlr_ptr
      type(c_ptr) :: tau_sampling_ptr, matsu_sampling_ptr
      integer(c_int), target :: basis_size, sve_size, max_size
      real(c_double), allocatable, target :: taus(:)
      integer(c_int64_t), allocatable, target :: matsus(:) ! Use int64_t for Matsubara indices
      real(c_double), allocatable, target :: poles(:)
      real(c_double), allocatable, target :: svals(:)

      print *, "Testing ", case_name, " case"

      ! Create a new kernel
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      if (status /= 0) then
         print *, "Error creating kernel"
         stop
      end if
      if (.not. c_associated(k_ptr)) then
         print *, "Error: kernel is not assigned"
         stop
      end if

      status = c_spir_kernel_domain(k_ptr, c_loc(xmin), c_loc(xmax), c_loc(ymin), c_loc(ymax))
      if (status /= 0) then
         print *, "Error: kernel domain is not assigned"
         stop
      end if
      print *, "Kernel domain =", xmin, xmax, ymin, ymax

      ! Create a copy of the kernel
      k_copy_ptr = c_spir_kernel_clone(k_ptr)

      ! Create a new SVE result
      print *, "Creating SVE result"

      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1.0_c_double, -1_c_int, -1_c_int, 1_c_int, c_loc(status))
      if (status /= 0) then
         print *, "Error creating SVE result"
         stop
      end if
      if (.not. c_associated(sve_ptr)) then
         print *, "Error: SVE result is not assigned"
         stop
      end if

      ! Get the size of the SVE result
      print *, "Getting SVE result size"
      status = c_spir_sve_result_get_size(sve_ptr, c_loc(sve_size))
      if (status /= 0) then
         print *, "Error getting SVE result size"
         stop
      end if

      ! Get the singular values of the SVE result
      ! Note: sve_size can be larger than the basis size.
      print *, "Getting SVE result singular values"
      allocate(svals(sve_size))
      status = c_spir_sve_result_get_svals(sve_ptr, c_loc(svals))
      if (status /= 0) then
         print *, "Error getting SVE result singular values"
         stop
      end if
      print *, "SVE result singular values =", svals

      ! Create a finite temperature basis
      print *, "Creating finite temperature basis"
      max_size = -1  ! Use default max size
      basis_ptr = c_spir_basis_new(statistics, beta, omega_max, k_ptr, sve_ptr, max_size, c_loc(status))
      if (status /= 0) then
         print *, "Error creating finite temperature basis"
         stop
      end if
      if (.not. c_associated(basis_ptr)) then
         print *, "Error: basis is not assigned"
         stop
      end if

      ! Get the size of the basis
      print *, "Getting basis size"
      status = c_spir_basis_get_size(basis_ptr, c_loc(basis_size))
      if (status /= 0) then
         print *, "Error getting basis size"
         stop
      end if
      print *, "Basis size =", basis_size

      ! Tau sampling
      print *, "Sampling tau"
      status = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntaus))
      if (status /= 0) then
         print *, "Error getting number of tau points"
         stop
      end if
      print *, "Number of tau points =", ntaus
      allocate(taus(ntaus))
      status = c_spir_basis_get_default_taus(basis_ptr, c_loc(taus))
      if (status /= 0) then
         print *, "Error getting tau points"
         stop
      end if
      print *, "Tau =", taus
      tau_sampling_ptr = c_spir_tau_sampling_new( &
         basis_ptr, ntaus, c_loc(taus), c_loc(status))
      if (status /= 0) then
         print *, "Error sampling tau"
         stop
      end if

      ! Matsubara sampling
      print *, "Sampling matsubara"
      status = c_spir_basis_get_n_default_matsus(basis_ptr, positive_only, c_loc(nmatsus))
      if (status /= 0) then
         print *, "Error getting number of matsubara points"
         stop
      end if
      print *, "Number of matsubara points =", nmatsus
      allocate(matsus(nmatsus))
      status = c_spir_basis_get_default_matsus(basis_ptr, positive_only, c_loc(matsus))
      if (status /= 0) then
         print *, "Error getting matsubara points"
         stop
      end if
      print *, "Matsubara =", matsus
      matsu_sampling_ptr = c_spir_matsu_sampling_new( &
         basis_ptr, positive_only, nmatsus, c_loc(matsus), c_loc(status))
      if (status /= 0) then
         print *, "Error sampling matsubara"
         stop
      end if

      ! Create a new DLR
      print *, "Creating DLR"
      print *, "basis_ptr is associated:", c_associated(basis_ptr)
      dlr_ptr = c_spir_dlr_new(basis_ptr, c_loc(status))
      print *, "After DLR creation - status:", status
      print *, "dlr_ptr is associated:", c_associated(dlr_ptr)
      if (status /= 0) then
         print *, "Error creating DLR"
         stop
      end if
      if (.not. c_associated(dlr_ptr)) then
         print *, "Error: DLR is not assigned"
         stop
      end if

      ! Get the number of poles
      print *, "Getting number of poles"
      print *, "Before get_npoles - dlr_ptr is associated:", c_associated(dlr_ptr)
      status = c_spir_dlr_get_npoles(dlr_ptr, c_loc(npoles))
      print *, "After get_npoles - status:", status
      print *, "After get_npoles - npoles:", npoles
      if (status /= 0) then
         print *, "Error getting number of poles"
         stop
      end if

      ! Get the poles
      print *, "Getting poles"
      print *, "Before get_poles - dlr_ptr is associated:", c_associated(dlr_ptr)
      print *, "Before get_poles - npoles:", npoles
      allocate(poles(npoles))
      print *, "After allocate poles"
      status = c_spir_dlr_get_poles(dlr_ptr, c_loc(poles))
      print *, "After get_poles - status:", status
      if (status /= 0) then
         print *, "Error getting poles"
         stop
      end if

      ! Test 2D operations
      call test_2d_operations(dlr_ptr, basis_ptr, tau_sampling_ptr, matsu_sampling_ptr, &
         npoles, basis_size, ntaus, nmatsus, poles, taus, matsus, epsilon)

      ! Deallocate
      deallocate(taus)
      deallocate(matsus)
      deallocate(poles)

      ! Release
      call c_spir_kernel_release(k_ptr)
      call c_spir_kernel_release(k_copy_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_basis_release(dlr_ptr)
      call c_spir_sampling_release(tau_sampling_ptr)
      call c_spir_sampling_release(matsu_sampling_ptr)
   end subroutine test_case

   subroutine test_2d_operations(dlr_ptr, basis_ptr, tau_sampling_ptr, matsu_sampling_ptr, &
      npoles, basis_size, ntaus, nmatsus, poles, taus, matsus, epsilon)
      type(c_ptr), intent(in) :: dlr_ptr, basis_ptr, tau_sampling_ptr, matsu_sampling_ptr
      integer(c_int), intent(in) :: npoles, basis_size, ntaus, nmatsus
      real(c_double), intent(in) :: poles(:), taus(:)
      integer(c_int64_t), intent(in) :: matsus(:)
      real(c_double), intent(in) :: epsilon

      ! Local variables
      real(c_double), allocatable, target :: coeffs(:,:), g_ir(:,:)
      complex(c_double), allocatable, target :: giw(:,:), giw_reconst(:,:), g_ir2_z(:,:), gtau_z(:,:)
      integer(c_int) :: status
      integer(c_int), target :: input_dims(2) = [0, 0]  ! Initialize with zeros
      integer(c_int), parameter :: target_dim = 0
      real(c_double), target :: tol
      integer(c_int), parameter :: order = SPIR_ORDER_COLUMN_MAJOR
      integer(c_int), parameter :: ndim = 2
      integer(c_int), parameter :: extra_size = 2

      ! Set tolerance
      tol = 10.0_c_double * epsilon

      ! Allocate arrays
      allocate(coeffs(npoles, extra_size))
      allocate(g_ir(basis_size, extra_size))
      allocate(g_ir2_z(basis_size, extra_size))
      allocate(gtau_z(ntaus, extra_size))
      allocate(giw(nmatsus, extra_size))
      allocate(giw_reconst(nmatsus, extra_size))

      ! Generate random coefficients
      call generate_random_coeffs(coeffs, poles, npoles, extra_size)

      ! Convert DLR coefficients to IR coefficients
      input_dims = [npoles, extra_size]
      status = c_spir_dlr2ir_dd(dlr_ptr, order, ndim, c_loc(input_dims), target_dim, &
         c_loc(coeffs), c_loc(g_ir))
      if (status /= 0) then
         print *, "Error converting DLR to IR"
         stop
      end if

      ! Evaluate Green's function at Matsubara frequencies from IR
      input_dims = [basis_size, extra_size]
      status = c_spir_sampling_eval_dz(matsu_sampling_ptr, order, ndim, &
         c_loc(input_dims), target_dim, &
         c_loc(g_ir), c_loc(giw))
      if (status /= 0) then
         print *, "Error evaluating Green's function at Matsubara frequencies"
         stop
      end if

      ! Convert Matsubara frequencies back to IR
      input_dims = [nmatsus, extra_size]
      status = c_spir_sampling_fit_zz(matsu_sampling_ptr, order, ndim, c_loc(input_dims), target_dim, &
         c_loc(giw), c_loc(g_ir2_z))
      if (status /= 0) then
         print *, "Error converting Matsubara frequencies back to IR"
         stop
      end if

      ! Compare IR coefficients (using real part of g_ir2_z)
      if (.not. compare_with_relative_error_d(g_ir, real(g_ir2_z), tol)) then
         print *, "Error: IR coefficients do not match after transformation cycle"
         stop
      end if

      ! Evaluate Green's function at tau points
      input_dims = [basis_size, extra_size]
      status = c_spir_sampling_eval_zz(tau_sampling_ptr, order, ndim, c_loc(input_dims), target_dim, &
         c_loc(g_ir2_z), c_loc(gtau_z))
      if (status /= 0) then
         print *, "Error evaluating Green's function at tau points"
         stop
      end if

      ! Convert tau points back to IR
      input_dims = [ntaus, extra_size]
      status = c_spir_sampling_fit_zz(tau_sampling_ptr, order, ndim, c_loc(input_dims), target_dim, &
         c_loc(gtau_z), c_loc(g_ir2_z))
      if (status /= 0) then
         print *, "Error converting tau points back to IR"
         stop
      end if

      ! Evaluate Green's function at Matsubara frequencies again
      input_dims = [basis_size, extra_size]
      status = c_spir_sampling_eval_zz(matsu_sampling_ptr, order, ndim, c_loc(input_dims), target_dim, &
         c_loc(g_ir2_z), c_loc(giw_reconst))
      if (status /= 0) then
         print *, "Error evaluating Green's function at Matsubara frequencies again"
         stop
      end if

      ! Compare the original and reconstructed Matsubara frequencies
      if (.not. compare_with_relative_error_z(giw, giw_reconst, tol)) then
         print *, "Error: Matsubara frequencies do not match after transformation cycle"
         stop
      end if

      ! Deallocate arrays
      deallocate(coeffs)
      deallocate(g_ir)
      deallocate(g_ir2_z)
      deallocate(gtau_z)
      deallocate(giw)
      deallocate(giw_reconst)
   end subroutine test_2d_operations

   function compare_with_relative_error_d(a, b, tol) result(is_close)
      real(c_double), intent(in) :: a(:,:), b(:,:)
      real(c_double), intent(in) :: tol
      logical :: is_close
      real(c_double) :: max_diff, max_ref
      integer :: i, j

      max_diff = 0.0_c_double
      max_ref = 0.0_c_double

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
      complex(c_double), intent(in) :: a(:,:), b(:,:)
      real(c_double), intent(in) :: tol
      logical :: is_close
      real(c_double) :: max_diff, max_ref
      integer :: i, j

      max_diff = 0.0_c_double
      max_ref = 0.0_c_double

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

   subroutine generate_random_coeffs(coeffs, poles, npoles, extra_size)
      real(c_double), intent(out) :: coeffs(:,:)
      real(c_double), intent(in) :: poles(:)
      integer(c_int), intent(in) :: npoles, extra_size
      integer :: i, j
      real(c_double) :: r

      do i = 1, npoles
         do j = 1, extra_size
            call random_number(r)
            coeffs(i,j) = (2.0_c_double * r - 1.0_c_double) * sqrt(abs(poles(i)))
         end do
      end do
   end subroutine generate_random_coeffs

end program
