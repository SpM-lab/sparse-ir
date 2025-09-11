#!/usr/bin/env python3
import re
import sys
from pathlib import Path


def extract_fortran_code_blocks(markdown_text):
    pattern = r'```(?:fortran|f90)\n(.*?)```'
    return re.findall(pattern, markdown_text, re.DOTALL)


def generate_test_file(code_blocks):
    test_code = '''program readme_samples_test
    implicit none

    print *, "Testing README code samples..."

'''

    # Add subroutine declarations
    for i in range(len(code_blocks)):
        test_code += f'''    call test_case_{i+1}()
'''

    test_code += '''
contains
'''

    # Add subroutines
    for i, block in enumerate(code_blocks):
        # Remove program/end program if present
        block = re.sub(r'program\s+\w+\s*$', '', block, flags=re.MULTILINE)
        block = re.sub(r'end\s+program\s+\w*\s*$', '', block, flags=re.MULTILINE)
        
        test_code += f'''
    subroutine test_case_{i+1}()
{block}
    end subroutine test_case_{i+1}
'''

    test_code += '''
end program readme_samples_test
'''
    return test_code


def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_fortran_test.py <readme_path> <output_path>")
        sys.exit(1)

    readme_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not readme_path.exists():
        print(f"Error: {readme_path} does not exist")
        sys.exit(1)

    with open(readme_path, 'r') as f:
        readme_text = f.read()

    code_blocks = extract_fortran_code_blocks(readme_text)
    if not code_blocks:
        print('No Fortran code blocks found in README.md')
        sys.exit(0)

    test_code = generate_test_file(code_blocks)
    with open(output_path, 'w') as f:
        f.write(test_code)


if __name__ == '__main__':
    main() 