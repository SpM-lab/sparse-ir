! Extends the SparseIR library with additional functionality
MODULE sparseir_ext
  USE, INTRINSIC :: iso_c_binding
  USE sparseir
  IMPLICIT NONE
  PRIVATE
  !
  PUBLIC :: IR, evaluate_tau, evaluate_matsubara, fit_tau, fit_matsubara, ir2dlr, dlr2ir
  PUBLIC :: init_ir, finalize_ir, eval_u_tau
  !
  INTEGER, PARAMETER :: DP = KIND(1.0D0)
  !
  !-----------------------------------------------------------------------
  TYPE IR
  !-----------------------------------------------------------------------
    !!
    !! This contains all the IR-basis objects,
    !! such as sampling points, and basis functions
    !!
    !
    INTEGER :: size
    !! total number of IR basis functions (size of s)
    INTEGER :: ntau
    !! total number of sampling points of imaginary time
    INTEGER :: nfreq_f
    !! total number of sampling Matsubara freqs (Fermionic)
    INTEGER :: nfreq_b
    !! total number of sampling Matsubara freqs (Bosonic)
    INTEGER :: nomega
    !! total number of sampling points of real frequency
    INTEGER :: npoles
    !! total number of DLR poles
    REAL(KIND=DP) :: beta
    !! inverse temperature
    REAL(KIND=DP) :: lambda
    !! lambda = 10^{nlambda},
    !! which determines maximum sampling point of real frequency
    REAL(KIND=DP) :: wmax
    !! maximum real frequency: wmax = lambda / beta
    REAL(KIND=DP) :: eps
    !! eps = 10^{-ndigit}
    REAL(KIND=DP), ALLOCATABLE :: s(:)
    !! singular values
    REAL(KIND=DP), ALLOCATABLE :: tau(:)
    !! sampling points of imaginary time
    REAL(KIND=DP), ALLOCATABLE :: omega(:)
    !! sampling points of real frequency
    INTEGER(KIND=8), ALLOCATABLE :: freq_f(:)
    !! integer part of sampling Matsubara freqs (Fermion)
    INTEGER(KIND=8), ALLOCATABLE :: freq_b(:)
    !! integer part of sampling Matsubara freqs (Boson)
    LOGICAL :: positive_only = .false.
    !! if true, take the Matsubara frequencies
    !! only from the positive region
    !
    TYPE(c_ptr) :: basis_f_ptr
    !! pointer to the fermionic basis
    TYPE(c_ptr) :: basis_b_ptr
    !! pointer to the bosonic basis
    type(c_ptr) :: sve_ptr
    !! pointer to the SVE result
    TYPE(c_ptr) :: k_ptr
    !! pointer to the kernel
    TYPE(c_ptr) :: tau_smpl_ptr
    !! pointer to the tau sampling points
    TYPE(c_ptr) :: matsu_f_smpl_ptr
    !! pointer to the fermionic frequency sampling points
    TYPE(c_ptr) :: matsu_b_smpl_ptr
    !! pointer to the bosonic frequency sampling points
    TYPE(c_ptr) :: dlr_f_ptr
    !! pointer to the fermionic DLR
    TYPE(c_ptr) :: dlr_b_ptr
    !! pointer to the bosonic DLR
  !-----------------------------------------------------------------------
   END TYPE IR
  !-----------------------------------------------------------------------
  !
  INTERFACE evaluate_tau
    MODULE PROCEDURE evaluate_tau_zz_1d, evaluate_tau_zz_2d, evaluate_tau_zz_3d, evaluate_tau_zz_4d, &
                     evaluate_tau_zz_5d, evaluate_tau_zz_6d, evaluate_tau_zz_7d, evaluate_tau_dd_1d, &
                     evaluate_tau_dd_2d, evaluate_tau_dd_3d, evaluate_tau_dd_4d, evaluate_tau_dd_5d, &
                     evaluate_tau_dd_6d, evaluate_tau_dd_7d
  END INTERFACE evaluate_tau
  !
  INTERFACE evaluate_matsubara
    MODULE PROCEDURE evaluate_matsubara_zz_1d, evaluate_matsubara_zz_2d, evaluate_matsubara_zz_3d, evaluate_matsubara_zz_4d, &
                     evaluate_matsubara_zz_5d, evaluate_matsubara_zz_6d, evaluate_matsubara_zz_7d, evaluate_matsubara_dz_1d, &
                     evaluate_matsubara_dz_2d, evaluate_matsubara_dz_3d, evaluate_matsubara_dz_4d, evaluate_matsubara_dz_5d, &
                     evaluate_matsubara_dz_6d, evaluate_matsubara_dz_7d
  END INTERFACE evaluate_matsubara
  !
  INTERFACE fit_tau
    MODULE PROCEDURE fit_tau_zz_1d, fit_tau_zz_2d, fit_tau_zz_3d, fit_tau_zz_4d, &
                     fit_tau_zz_5d, fit_tau_zz_6d, fit_tau_zz_7d, fit_tau_dz_1d, &
                     fit_tau_dz_2d, fit_tau_dz_3d, fit_tau_dz_4d, fit_tau_dz_5d, &
                     fit_tau_dz_6d, fit_tau_dz_7d, fit_tau_zd_1d, fit_tau_zd_2d, &
                     fit_tau_zd_3d, fit_tau_zd_4d, fit_tau_zd_5d, fit_tau_zd_6d, &
                     fit_tau_zd_7d, fit_tau_dd_1d, fit_tau_dd_2d, fit_tau_dd_3d, &
                     fit_tau_dd_4d, fit_tau_dd_5d, fit_tau_dd_6d, fit_tau_dd_7d
  END INTERFACE fit_tau
  !
  INTERFACE fit_matsubara
    MODULE PROCEDURE fit_matsubara_zz_1d, fit_matsubara_zz_2d, fit_matsubara_zz_3d, fit_matsubara_zz_4d, &
                     fit_matsubara_zz_5d, fit_matsubara_zz_6d, fit_matsubara_zz_7d, fit_matsubara_zd_1d, &
                     fit_matsubara_zd_2d, fit_matsubara_zd_3d, fit_matsubara_zd_4d, fit_matsubara_zd_5d, &
                     fit_matsubara_zd_6d, fit_matsubara_zd_7d
  END INTERFACE fit_matsubara
  !
  INTERFACE ir2dlr
    MODULE PROCEDURE ir2dlr_zz_1d, ir2dlr_zz_2d, ir2dlr_zz_3d, ir2dlr_zz_4d, &
                     ir2dlr_zz_5d, ir2dlr_zz_6d, ir2dlr_zz_7d, ir2dlr_dz_1d, &
                     ir2dlr_dz_2d, ir2dlr_dz_3d, ir2dlr_dz_4d, ir2dlr_dz_5d, &
                     ir2dlr_dz_6d, ir2dlr_dz_7d, ir2dlr_zd_1d, ir2dlr_zd_2d, &
                     ir2dlr_zd_3d, ir2dlr_zd_4d, ir2dlr_zd_5d, ir2dlr_zd_6d, &
                     ir2dlr_zd_7d, ir2dlr_dd_1d, ir2dlr_dd_2d, ir2dlr_dd_3d, &
                     ir2dlr_dd_4d, ir2dlr_dd_5d, ir2dlr_dd_6d, ir2dlr_dd_7d
  END INTERFACE ir2dlr
  !
  INTERFACE dlr2ir
    MODULE PROCEDURE dlr2ir_zz_1d, dlr2ir_zz_2d, dlr2ir_zz_3d, dlr2ir_zz_4d, &
                     dlr2ir_zz_5d, dlr2ir_zz_6d, dlr2ir_zz_7d, dlr2ir_dz_1d, &
                     dlr2ir_dz_2d, dlr2ir_dz_3d, dlr2ir_dz_4d, dlr2ir_dz_5d, &
                     dlr2ir_dz_6d, dlr2ir_dz_7d, dlr2ir_zd_1d, dlr2ir_zd_2d, &
                     dlr2ir_zd_3d, dlr2ir_zd_4d, dlr2ir_zd_5d, dlr2ir_zd_6d, &
                     dlr2ir_zd_7d, dlr2ir_dd_1d, dlr2ir_dd_2d, dlr2ir_dd_3d, &
                     dlr2ir_dd_4d, dlr2ir_dd_5d, dlr2ir_dd_6d, dlr2ir_dd_7d
  END INTERFACE dlr2ir
  !
  CONTAINS
  !
  FUNCTION create_logistic_kernel(lambda) RESULT(k_ptr)
    REAL(KIND=DP), INTENT(IN) :: lambda
    REAL(KIND=DP), TARGET :: lambda_c
    INTEGER(KIND=c_int), TARGET :: status_c
    TYPE(c_ptr) :: k_ptr
    lambda_c = lambda
    k_ptr = c_spir_logistic_kernel_new(lambda_c, c_loc(status_c))
  END FUNCTION create_logistic_kernel
  !
  FUNCTION create_sve_result(lambda, eps, k_ptr) RESULT(sve_ptr)
    REAL(KIND=DP), INTENT(IN) :: lambda
    REAL(KIND=DP), INTENT(IN) :: eps

    REAL(KIND=DP), TARGET :: lambda_c, eps_c
    REAL(KIND=DP) :: cutoff
    INTEGER(KIND=c_int), TARGET :: status_c

    TYPE(c_ptr), INTENT(IN) :: k_ptr
    TYPE(c_ptr) :: sve_ptr

    lambda_c = lambda
    eps_c = eps
    cutoff = -1.0
    sve_ptr = c_spir_sve_result_new(k_ptr, eps_c, cutoff, -1_c_int, -1_c_int, 1_c_int, c_loc(status_c))
    IF (status_c /= 0) THEN
       CALL errore('create_sve_result', 'Error creating SVE result', status_c)
    ENDIF
  END FUNCTION create_sve_result

  FUNCTION create_basis(statistics, beta, wmax, k_ptr, sve_ptr) RESULT(basis_ptr)
    INTEGER, INTENT(IN) :: statistics
    REAL(KIND=DP), INTENT(IN) :: beta
    REAL(KIND=DP), INTENT(IN) :: wmax
    TYPE(c_ptr), INTENT(IN) :: k_ptr, sve_ptr

    ! Function result type declaration
    TYPE(c_ptr) :: basis_ptr

    ! Local variable declarations
    INTEGER(KIND=c_int), TARGET :: max_size
    INTEGER(KIND=c_int), TARGET :: status_c
    INTEGER(KIND=c_int) :: statistics_c
    REAL(KIND=DP) :: beta_c, wmax_c

    ! Executable statements
    max_size = -1
    statistics_c = statistics
    beta_c = beta
    wmax_c = wmax
    basis_ptr = c_spir_basis_new(statistics_c, beta_c, wmax_c, k_ptr, sve_ptr, max_size, c_loc(status_c))
    IF (status_c /= 0) THEN
       CALL errore('create_basis', 'Error creating basis', status_c)
    ENDIF
  END FUNCTION create_basis

  FUNCTION get_basis_size(basis_ptr) RESULT(size)
    TYPE(c_ptr), INTENT(IN) :: basis_ptr
    INTEGER(KIND=c_int) :: size
    INTEGER(KIND=c_int), TARGET :: size_c
    INTEGER(KIND=c_int) :: status
    !
    status = c_spir_basis_get_size(basis_ptr, c_loc(size_c))
    IF (status /= 0) THEN
       CALL errore('get_basis_size', 'Error getting basis size', status)
    ENDIF
    size = size_c
  END FUNCTION get_basis_size

  FUNCTION basis_get_svals(basis_ptr) RESULT(svals)
    TYPE(c_ptr), INTENT(IN) :: basis_ptr
    REAL(KIND=DP), ALLOCATABLE :: svals(:)
    INTEGER(KIND=c_int), TARGET :: nsvals_c
    INTEGER(KIND=c_int) :: status
    REAL(KIND=DP), ALLOCATABLE, TARGET :: svals_c(:)
    !
    status = c_spir_basis_get_size(basis_ptr, c_loc(nsvals_c))
    IF (status /= 0) THEN
       call errore('basis_get_svals', 'Error getting number of singular values', status)
    ENDIF
    !
    ALLOCATE(svals_c(nsvals_c))
    !
    status = c_spir_basis_get_svals(basis_ptr, c_loc(svals_c))
    IF (status /= 0) THEN
       CALL errore('basis_get_svals', 'Error getting singular values', status)
    ENDIF
    !
    ALLOCATE(svals(nsvals_c))
    svals = svals_c
    !
    DEALLOCATE(svals_c)
  END FUNCTION basis_get_svals
  !
  SUBROUTINE create_tau_smpl(basis_ptr, tau, tau_smpl_ptr)
    TYPE(c_ptr), INTENT(IN) :: basis_ptr
    REAL(KIND=DP), ALLOCATABLE, INTENT(OUT) :: tau(:)
    TYPE(c_ptr), INTENT(OUT) :: tau_smpl_ptr
    !
    INTEGER(KIND=c_int), TARGET :: ntau_c
    INTEGER(KIND=c_int), TARGET :: status_c
    REAL(KIND=DP), ALLOCATABLE, TARGET :: tau_c(:)
    !
    status_c = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntau_c))
    IF (status_c /= 0) THEN
       CALL errore('create_tau_smpl', 'Error getting number of tau points', status_c)
    ENDIF
    !
    ALLOCATE(tau_c(ntau_c))
    IF (ALLOCATED(tau)) DEALLOCATE(tau)
    ALLOCATE(tau(ntau_c))
    !
    status_c = c_spir_basis_get_default_taus(basis_ptr, c_loc(tau_c))
    IF (status_c /= 0) THEN
       CALL errore('create_tau_smpl', 'Error getting tau points', status_c)
    ENDIF
    tau = REAL(tau_c, KIND=8)
    !
    tau_smpl_ptr = c_spir_tau_sampling_new(basis_ptr, ntau_c, c_loc(tau_c), c_loc(status_c))
    IF (status_c /= 0) THEN
       CALL errore('create_tau_smpl', 'Error creating tau sampling points', status_c)
    ENDIF
    !
    DEALLOCATE(tau_c)
  END SUBROUTINE create_tau_smpl
  !
  SUBROUTINE create_matsu_smpl(basis_ptr, positive_only, matsus, matsu_smpl_ptr)
    TYPE(c_ptr), INTENT(IN) :: basis_ptr
    LOGICAL, INTENT(IN) :: positive_only
    INTEGER(KIND=8), ALLOCATABLE, INTENT(OUT) :: matsus(:)
    TYPE(c_ptr), INTENT(OUT) :: matsu_smpl_ptr
    !
    INTEGER(KIND=c_int), TARGET :: nfreq_c
    INTEGER(KIND=c_int), TARGET :: status_c
    INTEGER(KIND=c_int64_t), ALLOCATABLE, TARGET :: matsus_c(:)
    INTEGER(KIND=c_int) :: positive_only_c
    !
    positive_only_c = MERGE(1, 0, positive_only)
    !
    status_c = c_spir_basis_get_n_default_matsus(basis_ptr, positive_only_c, c_loc(nfreq_c))
    IF (status_c /= 0) THEN
       CALL errore('create_matsu_smpl', 'Error getting number of fermionic frequencies', status_c)
    ENDIF
    !
    ALLOCATE(matsus_c(nfreq_c))
    !
    status_c = c_spir_basis_get_default_matsus(basis_ptr, positive_only_c, c_loc(matsus_c))
    IF (status_c /= 0) THEN
       CALL errore('create_matsu_smpl', 'Error getting frequencies', status_c)
    ENDIF
    !
    IF (ALLOCATED(matsus)) DEALLOCATE(matsus)
    ALLOCATE(matsus(nfreq_c))
    matsus = matsus_c
    !
    ! Create sampling object
    matsu_smpl_ptr = c_spir_matsu_sampling_new(basis_ptr, positive_only_c, nfreq_c, c_loc(matsus_c), c_loc(status_c))
    IF (status_c /= 0) THEN
       CALL errore('create_matsu_smpl', 'Error creating sampling object', status_c)
    ENDIF
    !
    DEALLOCATE(matsus_c)
  END SUBROUTINE create_matsu_smpl
  !
  FUNCTION basis_get_ws(basis_ptr) RESULT(ws)
    TYPE(c_ptr), INTENT(IN) :: basis_ptr
    REAL(KIND=DP), ALLOCATABLE :: ws(:)
    INTEGER(KIND=c_int), TARGET :: nomega_c
    INTEGER(KIND=c_int) :: status
    REAL(KIND=DP), ALLOCATABLE, TARGET :: ws_c(:)
    !
    status = c_spir_basis_get_n_default_ws(basis_ptr, c_loc(nomega_c))
    IF (status /= 0) THEN
       CALL errore('basis_get_ws', 'Error getting number of real frequencies', status)
    ENDIF
    !
    ALLOCATE(ws_c(nomega_c))
    !
    status = c_spir_basis_get_default_ws(basis_ptr, c_loc(ws_c))
    IF (status /= 0) THEN
       CALL errore('basis_get_ws', 'Error getting real frequencies', status)
    ENDIF
    !
    ALLOCATE(ws(nomega_c))
    ws = ws_c
    !
    DEALLOCATE(ws_c)
  END FUNCTION basis_get_ws
  !
  SUBROUTINE init_ir(obj, beta, lambda, eps, positive_only)
    !-----------------------------------------------------------------------
    !!
    !! This routine initializes arrays related to the IR-basis objects.
    !!
    !
    TYPE(IR), INTENT(INOUT) :: obj
    !! contains all the IR-basis objects
    REAL(KIND=DP), INTENT(IN) :: beta
    !! inverse temperature
    REAL(KIND=DP), INTENT(IN) :: lambda
    !! lambda = 10^{nlambda}
    REAL(KIND=DP), INTENT(IN) :: eps
    !! cutoff for the singular value expansion
    LOGICAL, INTENT(IN), OPTIONAL :: positive_only
    !! if true, take the Matsubara frequencies
    !! only from the positive region
    !
    LOGICAL :: lpositive
    REAL(KIND=DP) :: wmax
    INTEGER(KIND=c_int), TARGET :: status_c, npoles_c
    !
    TYPE(c_ptr) :: sve_ptr, basis_f_ptr, basis_b_ptr, k_ptr, dlr_f_ptr, dlr_b_ptr
    !
    wmax = lambda / beta
    write(*, *) 'wmax', wmax
    write(*, *) 'beta', beta
    lpositive = .false.
    IF (PRESENT(positive_only)) lpositive = positive_only
    !
    k_ptr = create_logistic_kernel(lambda)
    IF (.NOT. c_associated(k_ptr)) THEN
       CALL errore('init_ir', 'Kernel is not assigned', 1)
    ENDIF
    !
    sve_ptr = create_sve_result(lambda, eps, k_ptr)
    IF (.NOT. c_associated(sve_ptr)) THEN
       CALL errore('init_ir', 'SVE result is not assigned', 1)
    ENDIF
    !
    basis_f_ptr = create_basis(SPIR_STATISTICS_FERMIONIC, beta, wmax, k_ptr, sve_ptr)
    IF (.NOT. c_associated(basis_f_ptr)) THEN
       CALL errore('init_ir', 'Fermionic basis is not assigned', 1)
    ENDIF
    !
    basis_b_ptr = create_basis(SPIR_STATISTICS_BOSONIC, beta, wmax, k_ptr, sve_ptr)
    IF (.NOT. c_associated(basis_b_ptr)) THEN
       CALL errore('init_ir', 'Bosonic basis is not assigned', 1)
    ENDIF
    !
    ! Create DLR objects
    dlr_f_ptr = c_spir_dlr_new(basis_f_ptr, c_loc(status_c))
    IF (status_c /= 0 .OR. .NOT. c_associated(dlr_f_ptr)) THEN
       CALL errore('init_ir', 'Error creating fermionic DLR', status_c)
    ENDIF
    !
    dlr_b_ptr = c_spir_dlr_new(basis_b_ptr, c_loc(status_c))
    IF (status_c /= 0 .OR. .NOT. c_associated(dlr_b_ptr)) THEN
       CALL errore('init_ir', 'Error creating bosonic DLR', status_c)
    ENDIF
    !
    ! Get number of poles
    status_c = c_spir_dlr_get_npoles(dlr_f_ptr, c_loc(npoles_c))
    IF (status_c /= 0) THEN
       CALL errore('init_ir', 'Error getting number of poles', status_c)
    ENDIF
    !
    obj%positive_only = lpositive
    obj%beta = beta
    obj%wmax = wmax
    obj%lambda = lambda
    !
    obj%basis_f_ptr = basis_f_ptr
    obj%basis_b_ptr = basis_b_ptr
    obj%sve_ptr = sve_ptr
    obj%k_ptr = k_ptr
    obj%dlr_f_ptr = dlr_f_ptr
    obj%dlr_b_ptr = dlr_b_ptr
    obj%npoles = npoles_c
    !
    obj%size = get_basis_size(basis_f_ptr)
    !
    CALL create_tau_smpl(basis_f_ptr, obj%tau, obj%tau_smpl_ptr)
    !
    obj%s = basis_get_svals(basis_f_ptr)
    obj%ntau = size(obj%tau)
    CALL create_matsu_smpl(basis_f_ptr, lpositive, obj%freq_f, obj%matsu_f_smpl_ptr)
    CALL create_matsu_smpl(basis_b_ptr, lpositive, obj%freq_b, obj%matsu_b_smpl_ptr)
    obj%nfreq_f = size(obj%freq_f)
    obj%nfreq_b = size(obj%freq_b)
    obj%omega = basis_get_ws(basis_f_ptr)
    obj%nomega = size(obj%omega)
    obj%eps = eps
    !
  END SUBROUTINE init_ir
  !
  SUBROUTINE finalize_ir(obj)
    !-----------------------------------------------------------------------
    !!
    !! This routine deallocates IR-basis objects contained in obj
    !!
    !
    TYPE(IR), INTENT(INOUT) :: obj
    !! contains all the IR-basis objects
    INTEGER :: ierr
    !! Error status
    !
    ! Deallocate all member variables
    DEALLOCATE(obj%s, STAT = ierr)
    IF (ierr /= 0) CALL errore('finalize_ir', 'Error deallocating IR%s', 1)
    DEALLOCATE(obj%tau, STAT = ierr)
    IF (ierr /= 0) CALL errore('finalize_ir', 'Error deallocating IR%tau', 1)
    DEALLOCATE(obj%omega, STAT = ierr)
    IF (ierr /= 0) CALL errore('finalize_ir', 'Error deallocating IR%omega', 1)
    DEALLOCATE(obj%freq_f, STAT = ierr)
    IF (ierr /= 0) CALL errore('finalize_ir', 'Error deallocating IR%freq_f', 1)
    DEALLOCATE(obj%freq_b, STAT = ierr)
    IF (ierr /= 0) CALL errore('finalize_ir', 'Error deallocating IR%freq_b', 1)
    !-----------------------------------------------------------------------
    !
    IF (c_associated(obj%basis_f_ptr)) THEN
       CALL c_spir_basis_release(obj%basis_f_ptr)
    ENDIF
    IF (c_associated(obj%basis_b_ptr)) THEN
       call c_spir_basis_release(obj%basis_b_ptr)
    ENDIF
    IF (c_associated(obj%sve_ptr)) THEN
       CALL c_spir_sve_result_release(obj%sve_ptr)
    ENDIF
    IF (c_associated(obj%k_ptr)) THEN
       CALL c_spir_kernel_release(obj%k_ptr)
    ENDIF
    IF (c_associated(obj%tau_smpl_ptr)) THEN
       CALL c_spir_sampling_release(obj%tau_smpl_ptr)
    ENDIF
    IF (c_associated(obj%matsu_f_smpl_ptr)) THEN
       CALL c_spir_sampling_release(obj%matsu_f_smpl_ptr)
    ENDIF
    IF (c_associated(obj%matsu_b_smpl_ptr)) THEN
       CALL c_spir_sampling_release(obj%matsu_b_smpl_ptr)
    ENDIF
    IF (c_associated(obj%dlr_f_ptr)) THEN
       CALL c_spir_basis_release(obj%dlr_f_ptr)
    ENDIF
    IF (c_associated(obj%dlr_b_ptr)) THEN
       CALL c_spir_basis_release(obj%dlr_b_ptr)
    ENDIF
  END SUBROUTINE finalize_ir
  !
  !----------------------------------------------------------------------------
  SUBROUTINE errore( calling_routine, message, ierr )
  !----------------------------------------------------------------------------
  !
  ! ... This is a simple routine which writes an error message to output:
  ! ... if ierr <= 0 it does nothing,
  ! ... if ierr  > 0 it stops.
  !
  ! ...          **** Important note for parallel execution ***
  !
  ! ... in parallel execution unit 6 is written only by the first node;
  ! ... all other nodes have unit 6 redirected to nothing (/dev/null).
  ! ... We write to the "*" unit instead, that appears on all nodes.
  ! ... Effective but annoying!
  IMPLICIT NONE
  !
  CHARACTER(LEN=*), INTENT(IN) :: calling_routine, message
    ! the name of the calling calling_routine
    ! the output message
  INTEGER,          INTENT(IN) :: ierr
    ! the error flag
  INTEGER :: crashunit, mpime
  INTEGER, EXTERNAL :: find_free_unit
  CHARACTER(LEN=6) :: cerr
  !
  IF( ierr <= 0 ) RETURN
  !
  ! ... the error message is written on the "*" unit
  !
  WRITE( cerr, FMT = '(I6)' ) ierr
  WRITE( UNIT = *, FMT = '(/,1X,78("%"))' )
  WRITE( UNIT = *, FMT = '(5X,"Error in routine ",A," (",A,"):")' ) &
        TRIM(calling_routine), TRIM(ADJUSTL(cerr))
  WRITE( UNIT = *, FMT = '(5X,A)' ) TRIM(message)
  WRITE( UNIT = *, FMT = '(1X,78("%"),/)' )
  !
  WRITE( *, '("     stopping ...")' )
  !
  FLUSH( 5 )
  !
  STOP 1
  !
  RETURN
  !
  END SUBROUTINE errore

  FUNCTION eval_u_tau(obj, tau) RESULT(res)
    TYPE(IR), INTENT(IN) :: obj
    REAL(KIND=DP), INTENT(IN) :: tau
    !
    REAL(KIND=DP), ALLOCATABLE :: res(:)
    REAL(KIND=DP), ALLOCATABLE, TARGET :: res_c(:)
    INTEGER(KIND=c_int), TARGET :: status_c
    !
    TYPE(c_ptr) :: u_tau_ptr
    !
    u_tau_ptr = c_spir_basis_get_u(obj%basis_f_ptr, c_loc(status_c))
    IF (.NOT. c_associated(u_tau_ptr)) THEN
       CALL errore('eval_u_tau', 'Error getting u_tau pointer', status_c)
    END IF
    !
    ALLOCATE(res_c(obj%size))
    status_c = c_spir_funcs_eval(u_tau_ptr, tau, c_loc(res_c))
    IF (status_c /= 0) THEN
       CALL errore('eval_u_tau', 'Error evaluating u_tau', status_c)
    END IF
    !
    res = REAL(res_c, KIND=DP)
    !
    DEALLOCATE(res_c)
    CALL c_spir_funcs_release(u_tau_ptr)
  END FUNCTION eval_u_tau

  FUNCTION check_output_dims(target_dim, input_dims, output_dims) RESULT(is_valid)
    INTEGER, INTENT(IN) :: target_dim
    INTEGER(KIND=c_int), INTENT(IN) :: input_dims(:)
    INTEGER(KIND=c_int), INTENT(IN) :: output_dims(:)
    LOGICAL :: is_valid
    !
    integer :: i
    !
    IF (size(input_dims) /= size(output_dims)) THEN
       WRITE(*, *) "input_dims and output_dims have different sizes"
       STOP
    END IF
    !
    DO i = 1, size(input_dims)
       IF (i == target_dim) THEN
          CYCLE
       END IF
       IF (input_dims(i) /= output_dims(i)) THEN
          PRINT *, "input_dims(", i, ")", input_dims(i), "output_dims(", i, ")", output_dims(i)
          is_valid = .false.
          RETURN
       END IF
    END DO
    is_valid = .true.
  END FUNCTION check_output_dims
!
#define NAME evaluate_tau_zz_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define FLAT2 CMPLX(arr, kind=c_double)
#define RESHAPE_RES CMPLX(res_c, 0.0_DP, KIND=DP)
#define RESHAPE_RES2 CMPLX(res_cz, KIND=DP)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef FLAT2
#undef RESHAPE_RES
#undef RESHAPE_RES2
!
#define NAME evaluate_tau_zz_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define FLAT2 RESHAPE(CMPLX(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(CMPLX(res_c, 0.0_DP, KIND=DP), shape(res))
#define RESHAPE_RES2 RESHAPE(CMPLX(res_cz, KIND=DP), shape(res))
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_zz_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_zz_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_zz_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_zz_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_zz_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef FLAT2
#undef RESHAPE_RES
#undef RESHAPE_RES2
!
#define NAME evaluate_tau_dd_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define RESHAPE_RES REAL(res_c, KIND=DP)
#define DD
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DD
!
#define NAME evaluate_tau_dd_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(REAL(res_c, KIND=DP), shape(res))
#define DD
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_dd_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_dd_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_dd_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_dd_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_dd_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DD
!
#define NAME evaluate_tau_dz_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define RESHAPE_RES CMPLX(res_c, 0.0_DP, KIND=DP)
#define DZ
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DZ
!
#define NAME evaluate_tau_dz_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(CMPLX(res_c, 0.0_DP, KIND=DP), shape(res))
#define DZ
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_dz_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_dz_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_dz_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_dz_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_dz_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DZ
!
#define NAME evaluate_tau_zd_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define RESHAPE_RES REAL(res_c, KIND=DP)
#define ZD
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef ZD
!
#define NAME evaluate_tau_zd_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(REAL(res_c, KIND=DP), shape(res))
#define ZD
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_zd_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_zd_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_zd_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_zd_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_tau_zd_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "evaluate_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef ZD
!
#define NAME evaluate_matsubara_zz_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT CMPLX(arr, kind=c_double)
#define RESHAPE_RES CMPLX(res_c, KIND=DP)
#include "evaluate_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
!
#define NAME evaluate_matsubara_zz_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(CMPLX(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(CMPLX(res_c, KIND=DP), shape(res))
#include "evaluate_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_matsubara_zz_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "evaluate_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_matsubara_zz_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "evaluate_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_matsubara_zz_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "evaluate_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_matsubara_zz_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "evaluate_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_matsubara_zz_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "evaluate_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
!
#define NAME evaluate_matsubara_dz_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define RESHAPE_RES CMPLX(res_c, KIND=DP)
#define DZ
#include "evaluate_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DZ
!
#define NAME evaluate_matsubara_dz_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(CMPLX(res_c, KIND=DP), shape(res))
#define DZ
#include "evaluate_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_matsubara_dz_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "evaluate_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_matsubara_dz_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "evaluate_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_matsubara_dz_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "evaluate_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_matsubara_dz_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "evaluate_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME evaluate_matsubara_dz_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "evaluate_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DZ
!
#define NAME fit_tau_zz_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define FLAT2 CMPLX(arr, kind=c_double)
#define RESHAPE_RES CMPLX(res_c, 0.0_DP, KIND=DP)
#define RESHAPE_RES2 CMPLX(res_cz, KIND=DP)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef FLAT2
#undef RESHAPE_RES
#undef RESHAPE_RES2
!
#define NAME fit_tau_zz_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define FLAT2 RESHAPE(CMPLX(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(CMPLX(res_c, 0.0_DP, KIND=DP), shape(res))
#define RESHAPE_RES2 RESHAPE(CMPLX(res_cz, KIND=DP), shape(res))
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_zz_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_zz_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_zz_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_zz_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_zz_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef FLAT2
#undef RESHAPE_RES
#undef RESHAPE_RES2
!
#define NAME fit_tau_dz_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define RESHAPE_RES CMPLX(res_c, 0.0_DP, KIND=DP)
#define DZ
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DZ
!
#define NAME fit_tau_dz_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(CMPLX(res_c, 0.0_DP, KIND=DP), shape(res))
#define DZ
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_dz_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_dz_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_dz_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_dz_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_dz_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DZ
!
#define NAME fit_tau_zd_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define RESHAPE_RES REAL(res_c, KIND=DP)
#define ZD
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef ZD
!
#define NAME fit_tau_zd_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(REAL(res_c, KIND=DP), shape(res))
#define ZD
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_zd_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_zd_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_zd_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_zd_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_zd_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef ZD
!
#define NAME fit_tau_dd_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define RESHAPE_RES REAL(res_c, KIND=DP)
#define DD
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DD
!
#define NAME fit_tau_dd_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(REAL(res_c, KIND=DP), shape(res))
#define DD
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_dd_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_dd_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_dd_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_dd_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_tau_dd_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "fit_tau_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DD
!
#define NAME fit_matsubara_zz_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT CMPLX(arr, kind=c_double)
#define RESHAPE_RES CMPLX(res_c, KIND=DP)
#include "fit_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
!
#define NAME fit_matsubara_zz_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(CMPLX(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(CMPLX(res_c, KIND=DP), shape(res))
#include "fit_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_matsubara_zz_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "fit_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_matsubara_zz_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "fit_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_matsubara_zz_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "fit_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_matsubara_zz_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "fit_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_matsubara_zz_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "fit_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
!
#define NAME fit_matsubara_zd_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT CMPLX(arr, kind=c_double)
#define RESHAPE_RES REAL(res_c, KIND=DP)
#define ZD
#include "fit_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef ZD
!
#define NAME fit_matsubara_zd_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(CMPLX(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(REAL(res_c, KIND=DP), shape(res))
#define ZD
#include "fit_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_matsubara_zd_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "fit_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_matsubara_zd_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "fit_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_matsubara_zd_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "fit_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_matsubara_zd_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "fit_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME fit_matsubara_zd_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "fit_matsubara_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef ZD
!
#define NAME ir2dlr_zz_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define FLAT2 CMPLX(arr, kind=c_double)
#define RESHAPE_RES CMPLX(res_c, 0.0_DP, KIND=DP)
#define RESHAPE_RES2 CMPLX(res_cz, KIND=DP)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef FLAT2
#undef RESHAPE_RES
#undef RESHAPE_RES2
!
#define NAME ir2dlr_zz_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define FLAT2 RESHAPE(CMPLX(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(CMPLX(res_c, 0.0_DP, KIND=DP), shape(res))
#define RESHAPE_RES2 RESHAPE(CMPLX(res_cz, KIND=DP), shape(res))
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_zz_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_zz_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_zz_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_zz_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_zz_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef FLAT2
#undef RESHAPE_RES
#undef RESHAPE_RES2
!
#define NAME ir2dlr_dz_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define RESHAPE_RES CMPLX(res_c, 0.0_DP, KIND=DP)
#define DZ
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DZ
!
#define NAME ir2dlr_dz_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(CMPLX(res_c, 0.0_DP, KIND=DP), shape(res))
#define DZ
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_dz_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_dz_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_dz_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_dz_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_dz_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DZ
!
#define NAME ir2dlr_zd_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define RESHAPE_RES REAL(res_c, KIND=DP)
#define ZD
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef ZD
!
#define NAME ir2dlr_zd_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(REAL(res_c, KIND=DP), shape(res))
#define ZD
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_zd_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_zd_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_zd_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_zd_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_zd_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef ZD
!
#define NAME ir2dlr_dd_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define RESHAPE_RES REAL(res_c, KIND=DP)
#define DD
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DD
!
#define NAME ir2dlr_dd_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(REAL(res_c, KIND=DP), shape(res))
#define DD
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_dd_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_dd_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_dd_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_dd_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME ir2dlr_dd_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "ir2dlr_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DD
!
#define NAME dlr2ir_zz_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define FLAT2 CMPLX(arr, kind=c_double)
#define RESHAPE_RES CMPLX(res_c, 0.0_DP, KIND=DP)
#define RESHAPE_RES2 CMPLX(res_cz, KIND=DP)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef FLAT2
#undef RESHAPE_RES
#undef RESHAPE_RES2
!
#define NAME dlr2ir_zz_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define FLAT2 RESHAPE(CMPLX(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(CMPLX(res_c, 0.0_DP, KIND=DP), shape(res))
#define RESHAPE_RES2 RESHAPE(CMPLX(res_cz, KIND=DP), shape(res))
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_zz_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_zz_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_zz_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_zz_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_zz_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef FLAT2
#undef RESHAPE_RES
#undef RESHAPE_RES2
!
#define NAME dlr2ir_dz_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define RESHAPE_RES CMPLX(res_c, 0.0_DP, KIND=DP)
#define DZ
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DZ
!
#define NAME dlr2ir_dz_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(CMPLX(res_c, 0.0_DP, KIND=DP), shape(res))
#define DZ
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_dz_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_dz_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_dz_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_dz_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_dz_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DZ
!
#define NAME dlr2ir_zd_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define RESHAPE_RES REAL(res_c, KIND=DP)
#define ZD
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef ZD
!
#define NAME dlr2ir_zd_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(REAL(res_c, KIND=DP), shape(res))
#define ZD
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_zd_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_zd_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_zd_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_zd_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_zd_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef ZD
!
#define NAME dlr2ir_dd_1d
#define NDIM 1
#define SHAPE_ (:)
#define FLAT REAL(arr, kind=c_double)
#define RESHAPE_RES REAL(res_c, KIND=DP)
#define DD
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DD
!
#define NAME dlr2ir_dd_2d
#define NDIM 2
#define SHAPE_ (:,:)
#define FLAT RESHAPE(REAL(arr, kind=c_double), [size(arr)])
#define RESHAPE_RES RESHAPE(REAL(res_c, KIND=DP), shape(res))
#define DD
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_dd_3d
#define NDIM 3
#define SHAPE_ (:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_dd_4d
#define NDIM 4
#define SHAPE_ (:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_dd_5d
#define NDIM 5
#define SHAPE_ (:,:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_dd_6d
#define NDIM 6
#define SHAPE_ (:,:,:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
!
#define NAME dlr2ir_dd_7d
#define NDIM 7
#define SHAPE_ (:,:,:,:,:,:,:)
#include "dlr2ir_impl.fh"
#undef NAME
#undef NDIM
#undef SHAPE_
#undef FLAT
#undef RESHAPE_RES
#undef DD
!
END MODULE sparseir_ext