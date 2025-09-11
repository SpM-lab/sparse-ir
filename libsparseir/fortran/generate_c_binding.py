import clang.cindex
from clang.cindex import Index, TypeKind, CursorKind
import re

def count_pointer_depth(ctype):
    depth = 0
    while ctype.kind == TypeKind.POINTER:
        ctype = ctype.get_pointee()
        depth += 1
    return depth


def map_c_type_to_fortran(ctype):
    """Map a C type to a Fortran iso_c_binding type."""
    kind = ctype.kind
    if kind == TypeKind.VOID:
        return 'subroutine'
    elif kind == TypeKind.DOUBLE:
        return 'real(c_double), value'
    elif kind == TypeKind.INT or kind == TypeKind.INT:
        return 'integer(c_int), value'
    elif kind == TypeKind.UINT or kind == TypeKind.UINT:
        return 'integer(c_int), value'
    elif kind == TypeKind.POINTER:
        depth = count_pointer_depth(ctype)
        if depth == 1:
            return 'type(c_ptr), value'
        elif depth == 2:
            return 'type(c_ptr)'
        else:
            raise ValueError(f"Unsupported pointer depth: {depth}")
    elif kind == TypeKind.COMPLEX:
        return 'complex(c_double_complex), value'
    elif kind == TypeKind.ENUM:
        return 'integer(c_int), value'
    elif kind == TypeKind.ELABORATED:
        type_name = ctype.get_canonical().spelling
        if type_name in ["int32_t", "int32_t"]:
            return 'integer(c_int), value'
        return 'type(c_ptr)'
    else:
        return 'type(c_ptr)'  # default fallback


def generate_fortran_interface(cursor, types):
    """Generate Fortran interface code from a C function declaration."""
    if cursor.kind != CursorKind.FUNCTION_DECL:
        return "", ""

    func_name = cursor.spelling
    # Skip functions that start with underscore
    # as these functions are not part of the public API
    if func_name.startswith('_'):
        return "", ""

    fortran_func_name = f"c_{func_name}"
    result_type = map_c_type_to_fortran(cursor.result_type)

    args = []
    decls = []
    fortran_args = []
    fortran_body = []
    for arg in cursor.get_arguments():
        name = arg.spelling or 'arg'
        ftype = map_c_type_to_fortran(arg.type)
        args.append(name)
        decls.append(f"  {ftype} :: {name}")
        
        # Convert C types to Fortran types for friendly interface
        if 'type(c_ptr)' in ftype:
            # Get the actual type from the C type
            pointee = arg.type.get_pointee()
            type_name = pointee.get_canonical().spelling
            
            # Clean up type name: remove 'const' and 'struct _'
            type_name = type_name.replace('const ', '')
            type_name = type_name.replace('struct _', '')
            
            # Convert C type name to Fortran type name
            if type_name.startswith('spir_'):
                type_name = type_name  # Already has spir_ prefix
            else:
                type_name = f"spir_{type_name}"
            
            if type_name in types:
                # For output parameters, use intent(inout)
                if 'intent(out)' in ftype:
                    fortran_args.append(
                        f"  type({type_name}), intent(inout) :: {name}"
                    )
                else:
                    fortran_args.append(
                        f"  type({type_name}), intent(in) :: {name}"
                    )
                fortran_body.append(f"  {name}%%ptr")  # Use %% to escape %
            else:
                fortran_args.append(f"  {ftype} :: {name}")
                fortran_body.append(f"  {name}")
        else:
            fortran_args.append(f"  {ftype} :: {name}")
            fortran_body.append(f"  {name}")

    arglist = ', '.join(args)
    decl_lines = '\n'.join(decls)
    fortran_decl_lines = '\n'.join(fortran_args)
    fortran_body_lines = ',\n'.join(fortran_body)

    # Generate C binding interface
    if result_type == 'subroutine':
        c_binding = f"""
subroutine {fortran_func_name}({arglist}) &
    bind(c, name="{func_name}")
  use iso_c_binding
{decl_lines}
end subroutine
""".strip()
    else:
        result_decl = f"  {result_type.split(',')[0]} :: {fortran_func_name}"
        c_binding = f"""
function {fortran_func_name}({arglist}) &
    bind(c, name="{func_name}") result({fortran_func_name})
  use iso_c_binding
{decl_lines}
{result_decl}
end function
""".strip()

    # Generate Fortran-friendly interface
    # Remove spir_ prefix for Fortran interface
    fortran_name = func_name[5:] if func_name.startswith('spir_') else func_name
    if result_type == 'subroutine':
        fortran_interface = f"""
subroutine {fortran_name}({arglist})
  use iso_c_binding
{fortran_decl_lines}
  call {fortran_func_name}({fortran_body_lines})
end subroutine
""".strip()
    else:
        fortran_interface = f"""
function {fortran_name}({arglist}) result({fortran_name})
  use iso_c_binding
{fortran_decl_lines}
  {fortran_name} = {fortran_func_name}({fortran_body_lines})
end function
""".strip()

    return c_binding, fortran_interface


def generate_fortran_type_definition(types):
    """Generate Fortran type definitions for C types."""
    content = []
    for type_name in types:
        fortran_type_name = f"{type_name}_handle"
        content.append(f"""  type :: {fortran_type_name}
    type(c_ptr) :: handle = c_null_ptr
  contains
    final :: spir_release_{type_name}
  end type""")
    return "\n\n".join(content)


def generate_fortran_type_implementation(types):
    """Generate Fortran type implementations."""
    content = []
    for type_name in types:
        fortran_type_name = f"{type_name}_handle"
        # remove spir_ prefix if present
        type_name_without_prefix = type_name.replace('spir_', '')
        content.append(f"""  ! Assignment operator implementation
  subroutine assign_{type_name}(lhs, rhs)
    type({fortran_type_name}), intent(inout) :: lhs
    type({fortran_type_name}), intent(in) :: rhs
    
    ! Check for self-assignment
    if (c_associated(lhs%handle, rhs%handle)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%handle)) then
      call c_spir_release_{type_name_without_prefix}(lhs%handle)
      lhs%handle = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%handle)) then
      lhs%handle = c_spir_clone_{type_name_without_prefix}(rhs%handle)
    end if
  end subroutine

  ! Finalizer for {type_name}
  subroutine spir_release_{type_name}(this)
    type({fortran_type_name}), intent(inout) :: this
    
    if (c_associated(this%handle)) then
      call c_spir_release_{type_name_without_prefix}(this%handle)
      this%handle = c_null_ptr
    end if
  end subroutine""")
    return "\n\n".join(content)



def find_c_types(header_path):
    """Find all types declared with DECLARE_OPAQUE_TYPE macro."""
    types = set()
    pattern = re.compile(r'DECLARE_OPAQUE_TYPE\(([a-zA-Z0-9_]+)\)')
    
    with open(header_path, 'r') as f:
        content = f.read()
    
    # Remove #define section to avoid matching the macro definition itself
    content = re.sub(r'#define.*?\\\n.*?\\\n.*?\\\n.*?\\\n', '', content, flags=re.DOTALL)
    
    # Find all matches
    for match in pattern.finditer(content):
        type_name = match.group(1)
        # Skip if the type is not declared with DECLARE_OPAQUE_TYPE
        if not f"DECLARE_OPAQUE_TYPE({type_name})" in content:
            continue
        # Add spir_ prefix to type name
        types.add(f"spir_{type_name}")
    
    # Print found types
    print("Found types declared with DECLARE_OPAQUE_TYPE:")
    for type_name in sorted(types):
        print(f"  - {type_name}")
    
    return sorted(list(types))


def generate_proc_inc(types):
    """Generate _proc.inc file content."""
    content = []
    
    # Generate is_initialized interface
    content.append("interface is_initialized")
    for type_name in types:
        content.append(f"""  ! Check if {type_name} is initialized
  module function {type_name}_is_initialized(this) result(initialized)
    class({type_name}), intent(in) :: this
    logical :: initialized
  end function""")
    content.append("end interface\n")

    # Generate clone interface
    content.append("interface clone")
    for type_name in types:
        content.append(f"""  ! Clone the {type_name} object (create a copy)
  module function {type_name}_clone(this) result(copy)
    class({type_name}), intent(in) :: this
    type({type_name}) :: copy
  end function""")
    content.append("end interface\n")

    # Generate assign interface
    content.append("interface assign")
    for type_name in types:
        content.append(f"""  ! Assignment operator implementation
  module subroutine {type_name}_assign(lhs, rhs)
    class({type_name}), intent(inout) :: lhs
    class({type_name}), intent(in) :: rhs
  end subroutine""")
    content.append("end interface\n")

    # Generate finalize interface
    content.append("interface finalize")
    for type_name in types:
        content.append(f"""  ! Finalizer for {type_name}
  module subroutine {type_name}_finalize(this)
    type({type_name}), intent(inout) :: this
  end subroutine""")
    content.append("end interface\n")

    # Generate is_assigned interface
    content.append("interface is_assigned")
    for type_name in types:
        content.append(f"""  ! Check if {type_name}'s shared_ptr is assigned
  ! (contains a valid C++ object)
  module function {type_name}_is_assigned(this) result(assigned)
    type({type_name}), intent(in) :: this
    logical :: assigned
  end function""")
    content.append("end interface\n")

    # Join with double newlines between interfaces
    return "\n\n".join(content)


def generate_fortran_assign_interfaces(types):
    """Generate Fortran assignment operator interfaces."""
    code = []
    code.append("! Assignment operator interfaces")
    code.append("interface assignment(=)")
    for type_name in types:
        if type_name.startswith("spir_"):
            code.append(f"  module procedure assign_{type_name}")
    code.append("end interface")
    return "\n".join(code)


def generate_cbinding_public(cursor):
    """Generate public declarations for C binding functions."""
    if cursor.kind != CursorKind.FUNCTION_DECL:
        return ""

    func_name = cursor.spelling
    fortran_func_name = f"c_{func_name}"
    return f"  public :: {fortran_func_name}"


def generate_fortran_types_public(types):
    """Generate public declarations for handle types."""
    content = []
    for type_name in types:
        fortran_type_name = f"{type_name}_handle"
        content.append(f"  public :: {fortran_type_name}")
    return "\n".join(content)


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generate_fortran_interfaces.py spir_sample.h")
        return

    header_path = sys.argv[1]
    index = Index.create()
    tu = index.parse(header_path, args=['-x', 'c', '-std=c99'])

    # Find C types first
    types = find_c_types(header_path)

    # Generate interface bindings
    c_interfaces = []
    c_public = []
    fortran_interfaces = []
    print("\nFound C functions:")
    for cursor in tu.cursor.get_children():
        if cursor.location.file and cursor.location.file.name == header_path:
            if cursor.kind == CursorKind.FUNCTION_DECL:
                print(f"  - {cursor.spelling}")
            c_binding, fortran_interface = generate_fortran_interface(cursor, types)
            if c_binding:
                c_interfaces.append(c_binding)
                c_public.append(generate_cbinding_public(cursor))
            if fortran_interface:
                fortran_interfaces.append(fortran_interface)

    # Write C binding interfaces
    with open("_cbinding.inc", 'w') as f:
        f.write("! Autogenerated Fortran interfaces for " + header_path + "\n")
        for code in c_interfaces:
            f.write(code + "\n\n")

    # Write C binding public declarations
    with open("_cbinding_public.inc", 'w') as f:
        f.write("! Autogenerated public declarations for C bindings\n")
        for code in c_public:
            f.write(code + "\n")

    # Write Fortran-friendly interfaces
    #with open("_fortran_funcs.inc", 'w') as f:
        #f.write("! Autogenerated Fortran-friendly interfaces\n")
        #for code in fortran_interfaces:
            #f.write(code + "\n\n")

    # Generate Fortran type definitions
    #fortran_types = generate_fortran_type_definition(types)
    #with open("_fortran_types.inc", "w") as f:
        #f.write(fortran_types)

    # Generate Fortran type public declarations
    #fortran_types_public = generate_fortran_types_public(types)
    #with open("_fortran_types_public.inc", "w") as f:
        #f.write("! Autogenerated public declarations for handle types\n")
        #f.write(fortran_types_public + "\n")

    # Generate Fortran type implementations
    #fortran_impl = generate_fortran_type_implementation(types)
    #with open("_impl_types.inc", "w") as f:
        #f.write(fortran_impl)
    
    print("\nGenerated Fortran interfaces written to _cbinding.inc")
    print("Generated Fortran-friendly interfaces written to _fortran_funcs.inc")
    print("Generated Fortran type definitions written to _fortran_types.inc")


if __name__ == "__main__":
    main()
