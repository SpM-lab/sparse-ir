! Fortran interface for the SparseIR library
! This module provides Fortran bindings to the SparseIR C API using iso_c_binding
module sparseir
   use, intrinsic :: iso_c_binding
   implicit none
   private

   ! Export public interfaces
   !include '_fortran_types_public.inc'
   include '_cbinding_public.inc'
   public :: SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_BOSONIC, SPIR_ORDER_COLUMN_MAJOR

   ! Constants for statistics types
   integer(c_int32_t), parameter :: SPIR_STATISTICS_FERMIONIC = 1_c_int32_t
   integer(c_int32_t), parameter :: SPIR_STATISTICS_BOSONIC = 0_c_int32_t
   integer(c_int32_t), parameter :: SPIR_ORDER_ROW_MAJOR = 0_c_int32_t
   integer(c_int32_t), parameter :: SPIR_ORDER_COLUMN_MAJOR = 1_c_int32_t

   ! Type definitions
   !include '_fortran_types.inc'

   ! Assignment operator interfaces
   !include '_fortran_assign.inc'

   ! C bindings
   interface
      include '_cbinding.inc'
   end interface

   !contains
   ! Type implementations
   !include '_impl_types.inc'

end module sparseir
